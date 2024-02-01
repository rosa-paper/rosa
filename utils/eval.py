""" Evaluate BLEU for FactorizedNet or LoraNet model

Example:
python eval_lorank.py evaltools.experiment=runs/e2e_nlg/e64_l1e-05_b32_f1.0_nsgd_m0.9_w0.01_nanone_nu100_namdistillgpt2_namelora_r0.1_leepoch_srandom_t0

"""
import os
import os.path as osp
import logging
import csv
import argparse
from tqdm import tqdm
import warnings
from itertools import groupby

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

from utils.pymteval import BLEUScore, NISTScore
import peftnet as pn
from utils.utils import load_object, get_ignore_list_e2e


def get_data(dataset_name, dataset_cache):
    # Load dataset
    assert dataset_name in ["eli5", "e2e_nlg"], "Dataset not supported"

    test_split = {"eli5": "validation_asks", "e2e_nlg": "test"}[dataset_name]

    test_dataset = load_dataset(
        dataset_name, split=test_split, cache_dir=dataset_cache
    ).flatten()
    return test_dataset


def evaluate_model(cmodel, test_dataset, tokenizer, device=None, batch_size=8,
                   output_path_preds=None, output_path_refs=None, compute_bleu=True):
    cmodel.eval()
    # Overwrite output files
    if output_path_preds is not None and os.path.exists(output_path_preds):
        os.remove(output_path_preds)

    if output_path_refs is not None and os.path.exists(output_path_refs):
        os.remove(output_path_refs)

    # Used to ensure that the number of predictions and references are equal
    num_mrs = 0
    num_preds = 0

    with torch.no_grad():
        logging.info("=> Testing model bleu scores (Device={}) ...".format(device))
        BLEU = BLEUScore()
        NIST = NISTScore()

        # Initialize model
        cmodel.to(device)
        model_fn = cmodel.module if isinstance(cmodel, nn.DataParallel) else cmodel
        if any([isinstance(model_fn, k) for k in [pn.RosaNet, pn.LoraNet, pn.IA3Net]]):
            model_fn = model_fn.peft_model
        else:
            model_fn = model_fn

        model_fn.eval()
        gen_cfg = GenerationConfig(
            no_repeat_ngram_size=4,
            num_beams=5,
            max_length=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Sort dataset by 'meaning_representation' to ensure all similar items are together
        sorted_dataset = sorted(test_dataset, key=lambda x: x['meaning_representation'])

        # Group the sorted dataset by `meaning_representation`
        grouped_data = [list(group) for key, group in
                        groupby(sorted_dataset, key=lambda x: x['meaning_representation'])]

        # Combine all references for each group
        grouped_data = [
            {
                "meaning_representation": group[0]['meaning_representation'],
                "human_reference": [item['human_reference'] for item in group]
            }
            for group in grouped_data
        ]

        num_pts = len(grouped_data)

        # Start of block where warnings are suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in tqdm(range(0, num_pts, batch_size)):
                batch = grouped_data[i:i + batch_size]

                input_strs = ["Input: {} Output: ".format(dp['meaning_representation']) for dp in batch]
                references = [item['human_reference'] for item in batch]

                inputs = tokenizer(
                    input_strs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                output_ids = model_fn.generate(
                    input_ids=inputs['input_ids'].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    generation_config=gen_cfg,
                )

                output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                output_strs = [
                    out.replace("Input: {} Output: ".format(dp['meaning_representation']), "").strip() for out, dp in
                    zip(output_strs, batch)
                ]

                if output_path_preds is not None and output_path_refs is not None:
                    # Append references to `output_path_refs`
                    with open(output_path_refs, "a", encoding='utf-8') as f:
                        writer = csv.writer(f)
                        for refs in references:
                            try:
                                rw = [[ref] for ref in refs]
                                writer.writerows(rw)
                            except:
                                raise ValueError("References must be a list of strings. Got refs: {}".format(refs))
                            writer.writerow([])
                            num_mrs += 1

                    with open(output_path_preds, "a", encoding='utf-8') as f:
                        writer = csv.writer(f)
                        for pred in output_strs:
                            writer.writerow([pred])
                            num_preds += 1

                assert num_mrs == num_preds, "Number of predictions and references must be equal"

                if compute_bleu:
                    for output_str, reference in zip(output_strs, references):
                        BLEU.append(output_str, reference)
                        NIST.append(output_str, reference)

        # Remove last newline from `output_path_refs`
        with open(output_path_refs, 'rb+') as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell() - 1
            while pos > 0 and f.read(1) != b"\n":
                pos -= 1
                f.seek(pos, os.SEEK_SET)
            if pos > 0:
                f.seek(pos, os.SEEK_SET)
                f.truncate()

        return {
            "bleu": BLEU.score() if compute_bleu else None,
            "nist": NIST.score() if compute_bleu else None,
        }


# https://stackoverflow.com/questions/76465343/huggingface-transformers-model-config-reported-this-is-a-deprecated-strategy-to
# @hydra.main(version_base=None, config_path="./", config_name="configs")
def evaluate_experiment(experiment_root, test_dataset, overwrite=False, min_records=620, all=False):
    experiment_args = load_object(osp.join(experiment_root, "args.pkl"))
    model_names = [name for name in os.listdir(experiment_root) if name.startswith("model_") and name.endswith(".pth")]

    if all:
        print("\t=> Evaluating all {} models".format(len(model_names)))
    else:
        model_names = ["model_best.pth"]
        print("\t=> Evaluating latest model only".format(len(model_names)))

    for model_name in model_names:
        output_path_refs = osp.join(experiment_root, "test_references.txt")
        output_filename = model_name.replace("model_", "test_predictions_").replace(".pth", ".txt")
        output_path_preds = osp.join(experiment_root, output_filename)

        # Check number of lines in output_path_preds
        if osp.exists(output_path_preds) and not overwrite:
            # ~/scratch/rosa/runs/e2e_nlg/e5_l0.0005_b10_f1.0_s512_mTrue_nadamw_mo0.9_w0.01_nalinear_nu500_namgpt2_namenone_r0.5_leepoch_sarandom_cTrue_t0
            with open(output_path_preds, "r") as f:
                num_lines = sum(1 for _ in f)
            if num_lines >= min_records:
                print(
                    "\t=> Skipping evaluation. {} already exists and has {} records".format(
                        output_path_preds, num_lines
                    ))
                continue

        # Define model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForCausalLM.from_pretrained(experiment_args['model']['name'])
        tokenizer = AutoTokenizer.from_pretrained(experiment_args['model']['name'])
        tokenizer.pad_token = tokenizer.eos_token

        # Factorize & Load pretrained model
        ignore_list = get_ignore_list_e2e(model) if experiment_args['train']['ignore_list'] else None
        cmodel = {
            "rosa": pn.RosaNet,
            "lora": pn.LoraNet,
            "ia3": pn.IA3Net,
            "none": lambda x, **kwargs: x
        }[experiment_args['fnmodel']['name'].lower()](
            model, ignore_list=ignore_list, **experiment_args['fnmodel']['params']
        )

        print("\t=> Loading model {} ...".format(model_name))
        print("\t=> Using device {}".format(device))
        dct_best = torch.load(osp.join(experiment_root, model_name))
        cmodel.load_state_dict(dct_best['model_state_dict'])
        cmodel.to(device)

        if experiment_args['dataset']['name'] != "e2e_nlg":
            raise NotImplementedError("Dataset {} not supported".format(experiment_args['dataset']['name']))

        evaluate_model(
            cmodel,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            device=device,
            output_path_preds=output_path_preds,
            output_path_refs=output_path_refs,
            compute_bleu=False,
        )


def main(args):
    test_dataset = get_data(args.dataset, args.cache)
    if args.experiment == '':
        experiments = [osp.join(args.root, d) for d in os.listdir(args.root) if osp.isdir(osp.join(args.root, d))]
        for i, experiment in enumerate(experiments):
            print("\n=> [{}/{}] Evaluating experiment {}".format(i + 1, len(experiments), experiment))
            evaluate_experiment(experiment, test_dataset=test_dataset, all=args.all)
    else:
        print("=> Generating predictions for experiment {}".format(args.experiment))
        evaluate_experiment(args.experiment, test_dataset=test_dataset, all=args.all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, default='', required=False, help='Experiment directory')
    parser.add_argument('-a', '--all', action='store_true', help='Evaluate all model.pth weights')
    parser.add_argument('-r', '--root', type=str, default='', help='Root directory of many experiments')
    parser.add_argument('-d', '--dataset', type=str, default='e2e_nlg', help='Dataset name')
    parser.add_argument('-c', '--cache', type=str,
                        help='Dataset cache directory')
    args = parser.parse_args()

    assert args.experiment != '' or args.root != '', "Either experiment or root must be specified"
    main(args)