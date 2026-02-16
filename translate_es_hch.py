#!/usr/bin/env python3
"""
Translate Spanish to Wixarika (Huichol) using a fine-tuned NLLB model.

Usage:
    # Translate a single sentence
    python translate_es_hch.py --checkpoint submission_3.pt --text "Hola, ¿cómo estás?"

    # Translate from a file (one sentence per line)
    python translate_es_hch.py --checkpoint submission_3.pt --input input.txt --output output.txt

    # Interactive mode
    python translate_es_hch.py --checkpoint submission_3.pt
"""

import argparse
import sys
import os

import sentencepiece as spm
import torch

# Add the local fairseq fork to the path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "fairseq"))

from fairseq import checkpoint_utils, tasks, utils
from fairseq.data import Dictionary, encoders

SRC_LANG = "spa_Latn"
TGT_LANG = "hch_Latn"
SPM_MODEL = os.path.join(REPO_ROOT, "NLLB-inference", "preprocess", "flores200_sacrebleu_tokenizer_spm.model")
DICT_PATH = os.path.join(REPO_ROOT, "NLLB-inference", "dictionary.txt")
LANGS_FILE = os.path.join(REPO_ROOT, "NLLB-inference", "langs_extra.txt")


def load_langs(langs_file):
    with open(langs_file) as f:
        return f.read().strip()


def build_generator(args, task, models):
    return task.build_generator(models, args)


def translate(text, sp, task, models, generator, src_lang, tgt_lang, beam=5, max_len_a=1.2, max_len_b=10):
    """Translate a single sentence."""
    # Tokenize with SentencePiece
    tokenized = sp.encode(text, out_type=str)
    # Prepend the source language token
    tokenized = [f"__{src_lang}__"] + tokenized

    # Encode to fairseq token IDs
    src_dict = task.source_dictionary
    tokens = [src_dict.index(t) for t in tokenized]
    tokens.append(src_dict.eos())
    src_tokens = torch.LongTensor(tokens).unsqueeze(0)

    src_lengths = torch.LongTensor([len(tokens)])

    if torch.cuda.is_available():
        src_tokens = src_tokens.cuda()
        src_lengths = src_lengths.cuda()

    sample = {
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        }
    }

    # Generate translation
    with torch.no_grad():
        hypos = generator.generate(models, sample)

    # Decode the best hypothesis
    best_hypo = hypos[0][0]
    hypo_tokens = best_hypo["tokens"].int().cpu()

    # Convert token IDs back to strings
    tgt_dict = task.target_dictionary
    hypo_str = tgt_dict.string(hypo_tokens, extra_symbols_to_ignore={tgt_dict.eos(), tgt_dict.pad()})

    # Remove target language token if present
    hypo_str = hypo_str.replace(f"__{tgt_lang}__", "").strip()

    # Detokenize with SentencePiece
    pieces = hypo_str.split()
    detokenized = sp.decode(pieces)

    # Remove potential dataset tag tokens
    detokenized = detokenized.replace("<MINED_DATA>", "").strip()

    return detokenized


def main():
    parser = argparse.ArgumentParser(description="Translate Spanish to Wixarika (Huichol)")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint (e.g., submission_3.pt)")
    parser.add_argument("--text", type=str, default=None, help="Text to translate")
    parser.add_argument("--input", type=str, default=None, help="Input file with one sentence per line")
    parser.add_argument("--output", type=str, default=None, help="Output file for translations")
    parser.add_argument("--beam", type=int, default=5, help="Beam size (default: 5)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    langs = load_langs(LANGS_FILE)

    # Set up fairseq args for loading the model
    task_args = argparse.Namespace(
        task="translation_multi_simple_epoch",
        langs=langs,
        lang_pairs=f"{SRC_LANG}-{TGT_LANG}",
        fixed_dictionary=DICT_PATH,
        source_lang=SRC_LANG,
        target_lang=TGT_LANG,
        encoder_langtok="src",
        decoder_langtok=True,
        sampling_method="temperature",
        sampling_temperature="1",
        data="",
        left_pad_source=True,
        left_pad_target=False,
    )

    task = tasks.setup_task(task_args)

    # Load the model
    print(f"Loading model from {args.checkpoint}...", file=sys.stderr)
    models, _model_args = checkpoint_utils.load_model_ensemble(
        [args.checkpoint],
        task=task,
    )

    for model in models:
        model.eval()
        if torch.cuda.is_available() and not args.cpu:
            model.cuda()
        model.half()

    # Build generator with beam search
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

    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(SPM_MODEL)

    def do_translate(text):
        return translate(text, sp, task, models, generator, SRC_LANG, TGT_LANG, beam=args.beam)

    # Translate based on input mode
    if args.text:
        result = do_translate(args.text)
        print(result)

    elif args.input:
        out_file = open(args.output, "w") if args.output else sys.stdout
        with open(args.input) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    print("", file=out_file)
                    continue
                result = do_translate(line)
                print(result, file=out_file)
                print(f"Translated line {i + 1}", file=sys.stderr)
        if args.output:
            out_file.close()

    else:
        # Interactive mode
        print("Interactive mode: type Spanish text and press Enter. Ctrl+D to exit.", file=sys.stderr)
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                result = do_translate(line)
                print(result)
                sys.stdout.flush()
        except EOFError:
            pass


if __name__ == "__main__":
    main()