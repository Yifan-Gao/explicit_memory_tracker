#!/usr/bin/env python
import os
import re
import torch
import string
import spacy
import json
from argparse import ArgumentParser
from tempfile import NamedTemporaryFile
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
import editdistance
from pytorch_pretrained_bert.tokenization import BertTokenizer
import itertools
import copy

MATCH_IGNORE = {'do', 'have', '?', 'is', 'are', 'did', 'was', 'does'}
FORCE=True
PUNCT_WORDS = set(string.punctuation)
IGNORE_WORDS = MATCH_IGNORE | PUNCT_WORDS
MAX_LEN = 300

LARGEMODEL = False
LOWERCASE = True
BERT_MODEL = 'pretrained_models/bert-base-uncased.tar.gz'
BERT_VOCAB = 'pretrained_models/bert-base-uncased-vocab.txt'
FILENAME = 'bu'

tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=LOWERCASE, cache_dir=None)
CLASSES = ['yes', 'no', 'irrelevant', 'more']


nlplg = spacy.load("en_core_web_lg")


def tokenize(doc, sent_split=False):
    # assume input contains multiple sentences
    if not doc.strip():
        return []
    tokenized_spacy = nlplg(doc)
    if sent_split:
        sents_of_tokens = []
        for sent in tokenized_spacy.sents:
            tokens = []
            for tok in sent:
                tokens.append({
                    'text': tok.text,
                    'text_with_ws': tok.text_with_ws,
                    'lemma': tok.lemma_,
                })
            sents_of_tokens.append(bert_tokenize(tokens))
        return sents_of_tokens
    else:
        tokens = []
        for tok in tokenized_spacy:
            tokens.append({
                'text': tok.text,
                'text_with_ws': tok.text_with_ws,
                'lemma': tok.lemma_,
            })
        return bert_tokenize(tokens)


def bert_tokenize(sent):
    tokens = []
    for i, t in enumerate(sent):
        subtokens = tokenizer.tokenize(t['text'].strip())  # filter out '\n'
        for st in subtokens:
            tokens.append({
                'text': t['text'],
                'text_with_ws': t['text_with_ws'],
                'lemma': t['lemma'],
                'sub': st,
                'text_id': i,
            })
    return tokens


def convert_to_ids(tokens):
    return tokenizer.convert_tokens_to_ids([t['sub'] for t in tokens])


def detokenize(tokens):
    words = []
    for i, t in enumerate(tokens):
        if t['text_id'] is None or (i and t['text_id'] == tokens[i-1]['text_id']):
            continue
        else:
            words.append(t['text_with_ws'])
    return ''.join(words)


def make_tag(tag):
    return {'text': tag, 'text_with_ws': tag, 'lemma': tag, 'sub': tag, 'text_id': tag}


def compute_metrics(preds, data):
    import evaluator
    with NamedTemporaryFile('w') as fp, NamedTemporaryFile('w') as fg:
        json.dump(preds, fp)
        fp.flush()
        json.dump([{'utterance_id': e['utterance_id'], 'answer': e['answer']} for e in data], fg)
        fg.flush()
        results = evaluator.evaluate(fg.name, fp.name, mode='combined')
        # results['combined'] = results['macro_accuracy'] * results['bleu_4']
        return results


def filter_answer(answer):
    return detokenize([a for a in answer if a['text'].lower() not in MATCH_IGNORE])


def filter_chunk(answer):
    return detokenize([a for a in answer if a['text'].lower() not in MATCH_IGNORE])


def get_span(context, answer):
    answer = filter_answer(answer)
    best, best_score = None, float('inf')
    stop = False
    for i in range(len(context)):
        if stop:
            break
        for j in range(i, len(context)):
            chunk = filter_chunk(context[i:j+1])
            if '\n' in chunk or '*' in chunk:  # do not extract span across sentences/bullets
                continue
            score = editdistance.eval(answer, chunk)
            if score < best_score or (score == best_score and j-i < best[1]-best[0]):
                best, best_score = (i, j), score
            if chunk == answer:
                stop = True
                break
    s, e = best
    while (not context[s]['text'].strip() or context[s]['text'] in PUNCT_WORDS) and s < e:
        s += 1
    while (not context[e]['text'].strip() or context[s]['text'] in PUNCT_WORDS) and s < e:
        e -= 1
    return s, e


def get_editdistance_score(q_t, t_snippet, snippet_start, snippet_end):
    question_filtered = filter_answer(q_t)
    span_filtered = filter_chunk(t_snippet[snippet_start:snippet_end+1])
    editdist = editdistance.eval(question_filtered, span_filtered)
    score = 1 - editdist / max(len(question_filtered), len(span_filtered))
    return score


def extract_span_E3(t_snippet, t_questions, questions):
    match = {}  # select the most related spans for each followup question, question-to-span mapping
    q2span_edit = {}
    for q, q_t in zip(questions, t_questions):
        start, end = get_span(t_snippet, q_t)
        # if span across two sentences
        if t_snippet[start]['idx'] == t_snippet[end]['idx']:
            pass
        elif t_snippet[start]['idx']+1 == t_snippet[end]['idx']:
            # print('+1')
            # keep the longest sub span in one sent
            for idx in range(len(t_snippet)-1):
                if (t_snippet[idx]['idx'] == t_snippet[start]['idx']) and (t_snippet[idx+1]['idx'] == t_snippet[end]['idx']):
                    pivot = idx
                    break
            if pivot - start + 1 > end - pivot:
                end = pivot
            else:
                start = pivot+1
            assert t_snippet[start]['idx'] == t_snippet[end]['idx']
        elif t_snippet[start]['idx']+2 == t_snippet[end]['idx']:
            for idx in range(len(t_snippet)-1):
                if (t_snippet[idx]['idx'] == t_snippet[start]['idx']) and (t_snippet[idx+1]['idx'] == t_snippet[start]['idx']+1):
                    pivot1 = idx
                    break
            for idx in range(len(t_snippet)-1):
                if (t_snippet[idx]['idx'] == t_snippet[start]['idx']+1) and (t_snippet[idx+1]['idx'] == t_snippet[end]['idx']):
                    pivot2 = idx
                    break
            span1_len = pivot1 - start + 1
            span2_len = pivot2 - pivot1
            span3_len = end - pivot2
            span_max_len = max([span1_len, span2_len, span3_len])
            if span1_len == span_max_len:
                end = pivot1
            elif span2_len == span_max_len:
                start = pivot1 + 1
                end = pivot2
            else:
                start = pivot2 + 1
            assert t_snippet[start]['idx'] == t_snippet[end]['idx']
            # print('+2 {}'.format(t_snippet[end]['idx'] - t_snippet[start]['idx']))
        else:
            raise NotImplementedError
        assert start <= end
        # save the editdistance, normalized by its upper bound (longer sequence)
        q2span_edit[q] = get_editdistance_score(q_t, t_snippet, start, end)
        match[q] = (start, end)
    return match, q2span_edit


def extract_clauses(data):
    title_segmenter = '\n\n'
    bullet_segmenter = '* '
    snippet = data['snippet'].replace("** ", "* ")
    if LOWERCASE:
        snippet = snippet.lower()
    # split title of rules:
    tmp_title_snippet = snippet.split(title_segmenter)
    if len(tmp_title_snippet) > 1:
        title = tmp_title_snippet[0].strip('#').strip()
        if title[:2] in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']:
            title = title[2:].strip()
        # title = '# {}'.format(title)
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
            next_start_pos = bullet_position[idx + 1] if idx + 1 < len(bullet_position) else len(context) + 1
            clauses_tokenized.append(tokenize(context[current_start_pos:next_start_pos].strip('\n').strip(), sent_split=False))
            clauses.append(context[current_start_pos:next_start_pos].strip('\n').strip())
    else:
        clauses_tokenized = tokenize(context.strip('\n').strip(), sent_split=True)
        clauses = [(detokenize(sent)).strip('\n').strip() for sent in clauses_tokenized]

    snippet_tokenized = [title_tokenized] + clauses_tokenized if title != '' else clauses_tokenized
    questions = data['questions']
    questions_tokenized = [tokenize(q, sent_split=False) for q in questions]

    # add sent_id for each sent
    for idx, clause in enumerate(snippet_tokenized):
        for token in clause:
            token.update({'idx': idx})

    snippet_tokenized_flat = list(itertools.chain.from_iterable(snippet_tokenized))

    ques2span, ques2span_editscore = extract_span_E3(snippet_tokenized_flat, questions_tokenized, questions)
    ques2sent = {}
    for ques in questions:
        span_s, span_e = ques2span[ques]
        sent_label = [0] * len(snippet_tokenized)
        for token in snippet_tokenized_flat[span_s:span_e+1]:
            sent_label[token['idx']] = 1
        assert sum(sent_label) == 1
        ques2sent[ques] = sent_label

    return {'snippet': [title] + clauses if title != '' else clauses,
            'snippet_t': snippet_tokenized, 'snippet_t_flat': snippet_tokenized_flat,
            'has_bullets': has_bullets, 'questions': {q: tq for q, tq in zip(questions, questions_tokenized)},
            'ques2sent': ques2sent, 'ques2span': ques2span, 'ques2span_editscore': ques2span_editscore}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', default='data/sharc/json/{}.json', help='directory for data')
    args = parser.parse_args()

    for split in ['dev', 'train']:
        fsplit = 'sharc_train' if split == 'train' else 'sharc_dev'
        with open(args.data.format(fsplit)) as f:
            data = json.load(f)
            ########################
            # construct tree mappings
            ########################
            ftree = 'data/trees_{}_{}.json'.format(FILENAME, split)
            if not os.path.isfile(ftree) or FORCE:
                tasks = {}
                for ex in data:
                    for h in ex['evidence']:
                        if 'followup_question' in h:
                            h['follow_up_question'] = h['followup_question']
                            h['follow_up_answer'] = h['followup_answer']
                            del h['followup_question']
                            del h['followup_answer']
                    if ex['tree_id'] in tasks:
                        task = tasks[ex['tree_id']]
                    else:
                        task = tasks[ex['tree_id']] = {'snippet': ex['snippet'], 'questions': set()}
                    for h in ex['history'] + ex['evidence']:
                        task['questions'].add(h['follow_up_question'])
                    if ex['answer'].lower() not in {'yes', 'no', 'irrelevant'}:
                        task['questions'].add(ex['answer'])
                keys = sorted(list(tasks.keys()))
                vals = [extract_clauses(tasks[k]) for k in tqdm(keys)]
                mapping = {k: v for k, v in zip(keys, vals)}
                with open(ftree, 'wt') as f:
                    json.dump(mapping, f, indent=2)
            else:
                with open(ftree) as f:
                    mapping = json.load(f)

            #######################
            # construct samples
            #######################
            fproc = 'data/proc_entail_{}_{}.pt'.format(FILENAME, split)
            stats = defaultdict(list)
            augment_data = []
            for ex in tqdm(data):
                if len(ex['evidence']) > 0:
                    augment_data.append(copy.deepcopy(ex))
                ex_answer = ex['answer'].lower()
                m = mapping[ex['tree_id']]
                ex['ann'] = a = {
                    'snippet_t': m['snippet_t'],
                    'snippet_t_flat': m['snippet_t_flat'],
                    'question': tokenize(ex['question'], sent_split=False),
                    'scenario': tokenize(ex['scenario'], sent_split=True),
                    'answer': tokenize(ex['answer'], sent_split=False),
                    'hanswer': [tokenize(h['follow_up_answer'], sent_split=False) for h in ex['history']],
                    'hquestion': [m['questions'][h['follow_up_question']] for h in ex['history']],
                }
                if ex_answer not in CLASSES:
                    a['answer_sent'] = m['ques2sent'][ex['answer']]
                    # calculate offset because [CLS] token in sequence
                    span_offset = a['answer_sent'].index(1) + 1
                    start, end = m['ques2span'][ex['answer']]
                    a['answer_span'] = (start + span_offset, end + span_offset)
                    a['answer_editscore'] = m['ques2span_editscore'][ex['answer']]
                else:
                    a['answer_sent'] = None
                    a['answer_span'] = None
                    a['answer_editscore'] = 0

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

                # check the correctness of span_offset
                if ex_answer not in CLASSES:
                    assert inp[a['answer_span'][0]:a['answer_span'][1]+1] == a['snippet_t_flat'][m['ques2span'][ex['answer']][0]:m['ques2span'][ex['answer']][1]+1]

                type_ids = [0] * len(inp)  # segment A

                # question answer history: 0 unknown; 1 yes; 2 no
                sent_gold_state = torch.LongTensor([0] * len(a['snippet_t']))
                for hqa in ex['history'] + ex['evidence']:
                    hq_sent = m['ques2sent'][hqa['follow_up_question']].index(1)
                    if hqa['follow_up_answer'].lower() == 'yes':
                        sent_gold_state[hq_sent] = 1
                    else:
                        sent_gold_state[hq_sent] = 2

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

                if ex_answer in CLASSES:
                    clf = CLASSES.index(ex_answer)
                    answer_sent = -1
                    answer_span_start = -1
                    answer_span_end = -1
                else:
                    clf = CLASSES.index('more')
                    answer_sent = a['answer_sent'].index(1)
                    answer_span_start, answer_span_end = a['answer_span']

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
                    'answer_class': clf,
                    'answer_sent': answer_sent,
                    'answer_span_start': answer_span_start,
                    'answer_span_end': answer_span_end,
                    'sent_state': sent_gold_state,
                    'answer_editscore': a['answer_editscore'],
                }

                stats['snippet_len'].append(len(ex['ann']['snippet_t_flat']))
                stats['scenario_len'].append(sum([len(scen) for scen in ex['ann']['scenario']]))
                stats['history_len'].append(sum([len(q) + 3 for q in ex['ann']['hquestion']]))
                stats['question_len'].append(len(ex['ann']['question']))
                stats['inp_len'].append(sum(input_mask))

            if split == 'train':
                for ex in tqdm(augment_data):
                    ex['utterance_id'] = "aug_{}".format(ex['utterance_id'])
                    ex['history'] = ex['evidence'] + ex['history']
                    ex['scenario'] = ''
                    ex_answer = ex['answer'].lower()
                    m = mapping[ex['tree_id']]
                    ex['ann'] = a = {
                        'snippet_t': m['snippet_t'],
                        'snippet_t_flat': m['snippet_t_flat'],
                        'question': tokenize(ex['question'], sent_split=False),
                        'scenario': tokenize(ex['scenario'], sent_split=True),
                        'answer': tokenize(ex['answer'], sent_split=False),
                        'hanswer': [tokenize(h['follow_up_answer'], sent_split=False) for h in ex['history']],
                        'hquestion': [m['questions'][h['follow_up_question']] for h in ex['history']],
                    }
                    if ex_answer not in CLASSES:
                        a['answer_sent'] = m['ques2sent'][ex['answer']]
                        # calculate offset because [CLS] token in sequence
                        span_offset = a['answer_sent'].index(1) + 1
                        start, end = m['ques2span'][ex['answer']]
                        a['answer_span'] = (start + span_offset, end + span_offset)
                        a['answer_editscore'] = m['ques2span_editscore'][ex['answer']]
                    else:
                        a['answer_sent'] = None
                        a['answer_span'] = None
                        a['answer_editscore'] = 0

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

                    # check the correctness of span_offset
                    if ex_answer not in CLASSES:
                        assert inp[a['answer_span'][0]:a['answer_span'][1] + 1] == a['snippet_t_flat'][
                                                                                   m['ques2span'][ex['answer']][0]:
                                                                                   m['ques2span'][ex['answer']][1] + 1]

                    type_ids = [0] * len(inp)  # segment A

                    # question answer history: 0 unknown; 1 yes; 2 no
                    sent_gold_state = torch.LongTensor([0] * len(a['snippet_t']))
                    for hqa in ex['history']:
                        hq_sent = m['ques2sent'][hqa['follow_up_question']].index(1)
                        if hqa['follow_up_answer'].lower() == 'yes':
                            sent_gold_state[hq_sent] = 1
                        else:
                            sent_gold_state[hq_sent] = 2

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

                    if ex_answer in CLASSES:
                        clf = CLASSES.index(ex_answer)
                        answer_sent = -1
                        answer_span_start = -1
                        answer_span_end = -1
                    else:
                        clf = CLASSES.index('more')
                        answer_sent = a['answer_sent'].index(1)
                        answer_span_start, answer_span_end = a['answer_span']

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
                        'answer_class': clf,
                        'answer_sent': answer_sent,
                        'answer_span_start': answer_span_start,
                        'answer_span_end': answer_span_end,
                        'sent_state': sent_gold_state,
                        'answer_editscore': a['answer_editscore'],
                    }

                    stats['snippet_len'].append(len(ex['ann']['snippet_t_flat']))
                    stats['scenario_len'].append(sum([len(scen) for scen in ex['ann']['scenario']]))
                    stats['history_len'].append(sum([len(q) + 3 for q in ex['ann']['hquestion']]))
                    stats['question_len'].append(len(ex['ann']['question']))
                    stats['inp_len'].append(sum(input_mask))
                print('augmeng {}, original {}'.format(len(augment_data), len(data)))
                data += augment_data

            for k, v in sorted(list(stats.items()), key=lambda tup: tup[0]):
                print(k)
                print('mean: {}'.format(sum(v) / len(v)))
                print('min: {}'.format(min(v)))
                print('max: {}'.format(max(v)))
            preds = [{'utterance_id': e['utterance_id'],
                      'answer': detokenize(e['feat']['inp'][e['feat']['answer_span_start']:e['feat']['answer_span_end']+1]) if e['feat']['answer_span_start'] != -1 else e['answer'].lower()} for
                     e in data]
            pprint(compute_metrics(preds, data))
            torch.save(data, fproc)




