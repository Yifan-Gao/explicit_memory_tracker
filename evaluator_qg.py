import os
from evaluator import MoreEvaluator, prepro
from argparse import ArgumentParser
from pprint import pprint
import string
import json
import numpy as np


def load_txt(loadpath):
    with open(loadpath, 'r', encoding='utf-8') as fh:
        print('Loading {}'.format(loadpath), end="   ...   ")
        file = fh.read().splitlines()
        print('Done!')
    return file


def evaluate_qg(gold, pred):
    gold_prepro = [prepro(sent) for sent in gold]
    pred_prepro = [prepro(sent) for sent in pred]
    evaluator = MoreEvaluator()
    results = evaluator.evaluate(gold_prepro, pred_prepro)
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file",
                        default="../fqg/output/20191122_default/model.10.bin.pred.e2e", help="output file to compare")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file",
                        default="save/20191122_model_kvmn_v1_identify_e2_identify_mode_separate_loss_span_weight_0.1_loss_identify_weight_0.1-identify/e2e_tgt.txt", help="target file")
    args = parser.parse_args()

    gold = load_txt(args.tgt_file)
    pred = load_txt(args.out_file)
    result = evaluate_qg(gold, pred)
    pprint(result)
    with open(os.path.join(args.out_file + '.json'), 'wt') as f:
        json.dump(result, f, indent=2)





