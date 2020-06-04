#!/usr/bin/env python
import os
import json
from argparse import ArgumentParser
from tqdm import tqdm
from preprocess_dm import detokenize, tokenizer, CLASSES
from collections import defaultdict


def save_txt(data, savepath):
    with open(savepath, 'w', encoding='utf-8') as f:
        print('Saving {}'.format(savepath), end="   ...   ")
        for item in data:
            f.write("%s\n" % item)
        print('Done!')


def create_split(trees):
    keys = sorted(list(trees.keys()))
    src, tgt = [], []
    stats = defaultdict(list)
    for k in tqdm(keys):
        v = trees[k]
        snippet = v['snippet_t']
        snippet_flat = v['snippet_t_flat']
        for q_str, q_tok in v['questions'].items():
            span = v['ques2span'][q_str]
            s, e = span
            span_text = detokenize(snippet_flat[s:e + 1])
            snippet_text = ' '.join([detokenize(sent_tok) for sent_tok in snippet])
            src_ = ' '.join([snippet_text, '[SEP]', span_text]).replace('\n', ' ').strip()
            tgt_ = q_str.replace('\n', ' ').strip()
            if src_ not in src and tgt_ not in tgt:
                src.append(src_)
                tgt.append(tgt_)
                stats['src'].append(len(tokenizer.tokenize(src_)))
                stats['tgt'].append(len(tokenizer.tokenize(tgt_)))
    return src, tgt, stats


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ftree', default='data/trees_bu_{}.json', help='path to tree mapping')
    parser.add_argument('--fqg', default='data/qg_{}_{}.txt', help='save path for gold file')
    parser.add_argument('--test', action='store_true', help='only preprocess prediction')
    parser.add_argument('--fpred', default='', help='path to pred span')
    parser.add_argument('--fgold', default='data/sharc/json/sharc_dev.json', help='gold dev path')
    args = parser.parse_args()

    if args.test:
        with open(args.fgold) as f:
            gold = json.load(f)
        with open(os.path.join(args.fpred, 'dev.preds.json')) as f:
            pred = json.load(f)
        tgt_qg_gold = []
        src_qg_pred = []
        tgt_e2e_gold = []
        src_e2e_pred = []
        for gold_i, pred_i in zip(gold, pred):
            assert gold_i['utterance_id'] == pred_i['utterance_id']
            if gold_i['answer'].lower() not in CLASSES:
                tgt_qg_gold.append(gold_i['answer'])
                src_pred_i = ' '.join([pred_i['snippet'], '[SEP]', pred_i['answer_span_text']]).replace('\n', ' ').strip()
                src_qg_pred.append(src_pred_i)
                if pred_i['answer'] not in CLASSES:
                    tgt_e2e_gold.append(gold_i['answer'])
                    src_e2e_pred.append(src_pred_i)
        fqg_src = os.path.join(args.fpred, 'qg_src.txt')
        fqg_tgt = os.path.join(args.fpred, 'qg_tgt.txt')
        fe2e_src = os.path.join(args.fpred, 'e2e_src.txt')
        fe2e_tgt = os.path.join(args.fpred, 'e2e_tgt.txt')
        save_txt(src_qg_pred, fqg_src)
        save_txt(tgt_qg_gold, fqg_tgt)
        save_txt(src_e2e_pred, fe2e_src)
        save_txt(tgt_e2e_gold, fe2e_tgt)
    else:
        for split in ['dev', 'train']:
            with open(args.ftree.format(split)) as f:
                trees = json.load(f)

            print('Flattening {}'.format(split))
            src, tgt, stats = create_split(trees)

            print('Total Num {} '.format(len(src),))
            for k, v in sorted(list(stats.items()), key=lambda tup: tup[0]):
                print(k)
                print('mean: {}'.format(sum(v) / len(v)))
                print('min: {}'.format(min(v)))
                print('max: {}'.format(max(v)))

            fsrc = os.path.join(args.fqg.format(split, 'src'))
            ftgt = os.path.join(args.fqg.format(split, 'tgt'))
            save_txt(src, fsrc)
            save_txt(tgt, ftgt)
