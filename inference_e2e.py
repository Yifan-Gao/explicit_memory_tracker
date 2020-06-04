import os
import re
import torch
import json
from pprint import pprint
from argparse import ArgumentParser
from dm.base_entail import Module
from preprocess_dm import tokenize, make_tag, convert_to_ids, MAX_LEN, compute_metrics, detokenize
from tqdm import tqdm
from qg.biunilm.decode_seq2seq import main as qg_s2s


def preprocess_qg(preds):
    data = []
    lines = []
    for pred in preds:
        if pred['answer'].lower() not in {'yes', 'no', 'irrelevant'}:
            src_pred_i = ' '.join([pred['snippet'], '[SEP]', pred['answer']]).replace('\n', ' ').strip()
            ex = {
                'utterance_id': pred['utterance_id'],
                'src': src_pred_i,
            }
            data.append(ex)
            lines.append(src_pred_i)
    return data, lines


def preprocess_snippet(data):
    title_segmenter = '\n\n'
    bullet_segmenter = '* '
    snippet = data['snippet'].replace("** ", "* ")
    snippet = snippet.lower()
    # split title of rules:
    tmp_title_snippet = snippet.split(title_segmenter)
    if len(tmp_title_snippet) > 1:
        title = tmp_title_snippet[0].strip('#').strip()
        if title[:2] in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']:
            title = title[2:].strip()
        title_tokenized = tokenize(title, sent_split=False)
        context = snippet[len(tmp_title_snippet[0]):].strip(title_segmenter)
    else:
        title = ''
        title_tokenized = []
        context = snippet
    # check if exist bullets
    has_bullets = False
    if bullet_segmenter in context:
        clauses = []
        clauses_tokenized = []
        has_bullets = True
        bullet_position = [m.start() for m in re.finditer('\* ', context)]
        if bullet_position[0] != 0:
            clauses_tokenized.append(tokenize(context[:bullet_position[0]].strip('\n').strip(), sent_split=False))
            clauses.append(context[:bullet_position[0]].strip('\n').strip())
        for idx in range(len(bullet_position)):
            current_start_pos = bullet_position[idx]
            next_start_pos = bullet_position[idx+1] if idx+1 < len(bullet_position) else len(context)+1
            clauses_tokenized.append(tokenize(context[current_start_pos:next_start_pos].strip('\n').strip(), sent_split=False))
            clauses.append(context[current_start_pos:next_start_pos].strip('\n').strip())
    else:
        clauses_tokenized = tokenize(context.strip('\n').strip(), sent_split=True)
        clauses = [(detokenize(sent)).strip('\n').strip() for sent in clauses_tokenized]

    snippet_tokenized = [title_tokenized] + clauses_tokenized if title != '' else clauses_tokenized

    return {'snippet': [title] + clauses if title != '' else clauses,
            'snippet_t': snippet_tokenized,
            'has_bullets': has_bullets, }


def preprocess(data):
    for ex in tqdm(data):
        m = preprocess_snippet(ex)
        ex['ann'] = a = {
            'snippet_t': m['snippet_t'],
            'question': tokenize(ex['question'], sent_split=False),
            'scenario': tokenize(ex['scenario'], sent_split=True),
            'hanswer': [tokenize(h['follow_up_answer'], sent_split=False) for h in ex['history']],
            'hquestion': [tokenize(h['follow_up_question'], sent_split=False) for h in ex['history']],
        }

        inp = []
        sep = make_tag('[SEP]')

        # snippets
        memory_idx = []  # representation position for key of memory network
        pointer_mask = []
        for clause in a['snippet_t']:
            if len(inp) < MAX_LEN: memory_idx.append(len(inp))
            inp += [make_tag('[CLS]')] + clause
            pointer_mask += ([0] + [1] * len(clause))  # [0] for [CLS]
        inp += [sep]

        type_ids = [0] * len(inp)  # segment A

        input_idx = []  # representation position for input of memory network
        # question
        if len(inp) < MAX_LEN: input_idx.append(len(inp))
        inp += [make_tag('[CLS]')] + a['question'] + [sep]
        # scenario (multiple sentence)
        for scenario in a['scenario']:
            if len(inp) < MAX_LEN: input_idx.append(len(inp))
            inp += [make_tag('[CLS]')] + scenario
        inp += [sep]
        # question answer history
        for hq, ha in zip(a['hquestion'], a['hanswer']):
            if len(inp) < MAX_LEN: input_idx.append(len(inp))
            inp += [make_tag('[CLS]')] + [make_tag('question')] + hq + [make_tag('answer')] + ha
        inp += [sep]

        type_ids += [1] * (len(inp) - len(type_ids))  # segment B
        pointer_mask += [0] * (len(inp) - len(pointer_mask))
        input_ids = convert_to_ids(inp)
        input_mask = [1] * len(inp)  # attention mask

        if len(inp) > MAX_LEN:
            inp = inp[:MAX_LEN]
            input_mask = input_mask[:MAX_LEN]
            type_ids = type_ids[:MAX_LEN]
            input_ids = input_ids[:MAX_LEN]
            pointer_mask = pointer_mask[:MAX_LEN]
        pad = make_tag('pad')
        while len(inp) < MAX_LEN:
            inp.append(pad)
            input_mask.append(0)
            type_ids.append(0)
            input_ids.append(0)
            pointer_mask.append(0)

        assert len(inp) == len(input_mask) == len(type_ids) == len(input_ids) == len(pointer_mask)

        ex['feat'] = {
            'inp': inp,
            'input_ids': torch.LongTensor(input_ids),
            'type_ids': torch.LongTensor(type_ids),
            'input_mask': torch.LongTensor(input_mask),
            'pointer_mask': torch.LongTensor(pointer_mask),
            'memory_idx': torch.LongTensor(memory_idx),
            'input_idx': torch.LongTensor(input_idx),
        }
    return data


def merge_edits(preds, qgpreds):
    # note: this happens in place
    qg = {p['utterance_id']: p for p in qgpreds}
    for p in preds:
        p['orig_answer'] = p['answer']
        if p['utterance_id'] in qg:
            p['answer'] = qg[p['utterance_id']]['tgt']
    return preds


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fin', default='data/sharc/json/sharc_dev.json', help='input data file')
    parser.add_argument('--dm', default='/opt/models/dm.pt', help='sharc model to use')
    parser.add_argument('--device', default='cuda', help='cpu not supported')
    parser.add_argument('--model_bert_base_path', default='', help='bert model to use')
    # copy from unilm
    parser.add_argument("--bert_model", default='bert-large-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--model_recover_path", default='/opt/models/qg.bin', type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--cache_path", default='/opt/models', type=str,
                        help="Yifan added, bert vocab path")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', default=True,
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=48,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--not_predict_token', type=str, default=None,
                        help="Do not predict the tokens during decoding.")

    args = parser.parse_args()

    print('loading raw file ...')
    with open(args.fin) as f:
        raw = json.load(f)

    print('preprocessing data')
    data = preprocess(raw)

    print('resuming dm from ' + args.dm)
    args_overwrite_dm = {
        'bert_model_path': args.model_bert_base_path,
        'model': 'c2f_entail',
        'dev_batch': args.batch_size,
        'device': args.device,
    }
    dm = Module.load(args.dm, override_args=args_overwrite_dm)
    dm.device = args.device
    dm.to(dm.device)
    dm.args.dev_batch = 10
    dm_preds = dm.run_pred_leaderboard(data)
    print("dm_preds {}".format(len(dm_preds)))

    qg_data, input_lines = preprocess_qg(dm_preds)
    print('qg_data {}, input_lines {}'.format(len(qg_data), len(input_lines)))
    output_lines = qg_s2s(opt=args, inputs=input_lines)
    print("output_lines {}".format(len(output_lines)))
    qg_preds = []
    for ex, input_line, output_line in zip(qg_data, input_lines, output_lines):
        assert ex['src'] == input_line
        ex['tgt'] = output_line
        qg_preds.append(ex)
    e2e_preds = merge_edits(dm_preds, qg_preds)

    metrics = compute_metrics(e2e_preds, raw)
    pprint(metrics)

