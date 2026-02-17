#!/usr/bin/env python3
"""
Translate dev.es using the fine-tuned NLLB model, save translations,
and compare against ground truth dev.hch.

Usage:
    python evaluate_dev.py --checkpoint submission_3.pt
    python evaluate_dev.py --checkpoint submission_3.pt --beam 5 --cpu
"""

import argparse
import sys
import os
from difflib import SequenceMatcher

import sacrebleu
import sentencepiece as spm
import torch

# Reuse everything from translate_es_hch
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "fairseq"))

from translate_es_hch import (
    load_langs,
    translate,
    build_generator,
    SRC_LANG,
    TGT_LANG,
    SPM_MODEL,
    DICT_PATH,
    LANGS_FILE,
)
from fairseq import checkpoint_utils, tasks

DATA_DIR = os.path.join(REPO_ROOT, "data")
DEV_ES = os.path.join(DATA_DIR, "dev.es")
DEV_HCH = os.path.join(DATA_DIR, "dev.hch")
OUTPUT_FILE = os.path.join(DATA_DIR, "dev.hch.pred")
REPORT_FILE = os.path.join(DATA_DIR, "dev_eval_report.txt")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ES→HCH translation on dev set")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--beam", type=int, default=5, help="Beam size (default: 5)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--match-threshold", type=float, default=0.8,
                        help="Similarity ratio threshold for SUCCESS (default: 0.8 = 80%%)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only translate the first N sentences (for quick testing)")
    args = parser.parse_args()

    # ── Load model (same setup as translate_es_hch.py) ──────────────────
    langs = load_langs(LANGS_FILE)

    task_args = argparse.Namespace(
        task="translation_multi_simple_epoch",
        langs=langs,
        lang_dict=None,
        lang_pairs=f"{SRC_LANG}-{TGT_LANG}",
        fixed_dictionary=DICT_PATH,
        source_dict=None,
        target_dict=None,
        source_lang=SRC_LANG,
        target_lang=TGT_LANG,
        encoder_langtok="src",
        decoder_langtok=True,
        lang_tok_replacing_bos_eos=False,
        sampling_method="temperature",
        sampling_temperature="1",
        data="",
        left_pad_source=True,
        left_pad_target=False,
        langtoks=None,
        langtoks_specs=["main"],
        extra_data=None,
        extra_lang_pairs=None,
        lang_tok_style="multilingual",
        add_data_source_prefix_tags=True,
        add_ssl_task_tokens=False,
        finetune_dict_specs=None,
        shuffle_instance=False,
        virtual_epoch_size=None,
        virtual_data_size=None,
        eval_lang_pairs=None,
        seed=1,
        pad_to_fixed_length=False,
    )

    task = tasks.setup_task(task_args)

    print(f"Loading model from {args.checkpoint}...", file=sys.stderr)
    models, _ = checkpoint_utils.load_model_ensemble(
        [args.checkpoint], task=task,
    )
    for model in models:
        model.eval()
        if torch.cuda.is_available() and not args.cpu:
            model.cuda()
        model.half()

    gen_args = argparse.Namespace(
        beam=args.beam,
        max_len_a=1.2,
        max_len_b=10,
        min_len=1,
        unnormalized=False,
        lenpen=1.0,
        unkpen=0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        sampling=False,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
    )
    generator = build_generator(gen_args, task, models)

    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    # ── Read dev data ───────────────────────────────────────────────────
    with open(DEV_ES) as f:
        src_lines = [line.strip() for line in f]
    with open(DEV_HCH) as f:
        ref_lines = [line.strip() for line in f]

    assert len(src_lines) == len(ref_lines), (
        f"Mismatch: dev.es has {len(src_lines)} lines, dev.hch has {len(ref_lines)}"
    )

    # Optionally limit to first N sentences
    if args.limit is not None:
        src_lines = src_lines[:args.limit]
        ref_lines = ref_lines[:args.limit]

    total = len(src_lines)
    matches = 0
    predictions = []
    threshold = args.match_threshold

    # ── Translate & compare ─────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as pred_f, open(REPORT_FILE, "w") as rep_f:
        rep_f.write(f"{'idx':>5}  {'status':>7}  {'sim':>6}  {'source / reference / prediction'}\n")
        rep_f.write("=" * 110 + "\n")

        for i, (src, ref) in enumerate(zip(src_lines, ref_lines)):
            if not src:
                pred = ""
            else:
                pred = translate(
                    src, sp, task, models, generator,
                    SRC_LANG, TGT_LANG, beam=args.beam,
                )

            predictions.append(pred)
            pred_f.write(pred + "\n")

            # Similarity ratio (0.0–1.0) via SequenceMatcher
            similarity = SequenceMatcher(None, ref.strip(), pred.strip()).ratio()
            is_match = similarity >= threshold
            if is_match:
                matches += 1
            status = "SUCCESS" if is_match else "FAIL"

            # Write to report
            rep_f.write(f"{i + 1:>5}  {status:>7}  {similarity:>5.1%}  SRC: {src}\n")
            rep_f.write(f"{'':>5}  {'':>7}  {'':>6}  REF: {ref}\n")
            rep_f.write(f"{'':>5}  {'':>7}  {'':>6}  PRED: {pred}\n")
            rep_f.write("-" * 110 + "\n")

            # Progress on stderr
            mark = "✓" if is_match else "✗"
            print(f"[{i + 1:>4}/{total}] {mark} ({similarity:.0%})  {src[:60]}", file=sys.stderr)

        # ── Corpus-level metrics (sacrebleu) ────────────────────────────
        bleu = sacrebleu.corpus_bleu(predictions, [ref_lines])
        chrf = sacrebleu.corpus_chrf(predictions, [ref_lines], word_order=0)   # chrF
        chrfpp = sacrebleu.corpus_chrf(predictions, [ref_lines], word_order=2)  # chrF++

        # ── Summary ─────────────────────────────────────────────────────
        accuracy = matches / total * 100 if total else 0
        summary = (
            f"\n{'=' * 110}\n"
            f"SUMMARY\n"
            f"  Similarity threshold : {threshold:.0%}\n"
            f"  Matches (≥ thresh)   : {matches}/{total}  ({accuracy:.2f}%)\n"
            f"\n"
            f"  BLEU   : {bleu.score:.2f}  ({bleu})\n"
            f"  chrF   : {chrf.score:.2f}  ({chrf})\n"
            f"  chrF++ : {chrfpp.score:.2f}  ({chrfpp})\n"
        )
        rep_f.write(summary)
        print(summary, file=sys.stderr)

    print(f"\nPredictions saved to: {OUTPUT_FILE}", file=sys.stderr)
    print(f"Evaluation report saved to: {REPORT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
