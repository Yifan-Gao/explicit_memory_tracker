import torch
from dm.base_entail import Module as Base
from torch import nn
from torch.nn import functional as F
from preprocess_dm import detokenize
from preprocess_dm import CLASSES
import collections


class Module(Base):

    def __init__(self, args, device='cpu'):
        super().__init__(args, device)
        self.top_k = args.span_top_k
        self.span_scorer = nn.Linear(self.args.bert_hidden_size, 2, bias=True)

    def forward(self, batch):
        out = super().forward(batch)
        span_scores = self.span_scorer(self.dropout(out['bert_enc']))
        out['span_scores'] = self.mask_scores(span_scores, out['pointer_mask'])
        return out

    def get_top_k(self, probs, k, sent_start, sent_end):
        p = list(enumerate(probs.tolist()))[sent_start : sent_end]
        p.sort(key=lambda tup: tup[1], reverse=True)
        return p[:k]

    def mask_scores(self, scores, mask):
        invalid = 1 - mask
        scores -= invalid.unsqueeze(2).expand_as(scores).float().mul(1e20)
        return scores

    def extract_preds(self, out, batch):
        preds = super().extract_preds(out, batch)
        scores = out['span_scores']
        ystart, yend = scores.split(1, dim=-1)
        pstart = F.softmax(ystart.squeeze(-1), dim=1)
        pend = F.softmax(yend.squeeze(-1), dim=1)

        for pred_i, pstart_i, pend_i, ex, pointer_mask_i in zip(preds, pstart, pend, batch, out['pointer_mask']):
            pred_i['answer_span_start'] = -1
            pred_i['answer_span_end'] = -1
            pred_i['answer_span_text'] = None
            if (pred_i['answer'] not in ['yes', 'no', 'irrelevant']) or (CLASSES[ex['feat']['answer_class']] == 'more'):
                # start inclusive, end exclusive
                sentence_start = ex['feat']['memory_idx'].to(self.device) + 1  # +1 to skip the cls token, inclusive
                sentence_end = torch.cat([ex['feat']['memory_idx'][1:].to(self.device), (pointer_mask_i == 1).nonzero()[-1] + 1])  # exclusive
                top_preds = []
                for sent_s, sent_e in zip(sentence_start, sentence_end):
                    top_start = self.get_top_k(pstart_i, self.top_k, sent_s, sent_e)
                    top_end = self.get_top_k(pend_i, self.top_k, sent_s, sent_e)
                    for s, ps in top_start:
                        for e, pe in top_end:
                            if e >= s:
                                top_preds.append((s, e, ps * pe))
                top_preds = sorted(top_preds, key=lambda tup: tup[-1], reverse=True)[:self.top_k]
                top_answers = [(detokenize(ex['feat']['inp'][s:e + 1]), s, e, p) for s, e, p in top_preds]
                top_ans, top_s, top_e, top_p = top_answers[0]
                # pred_i['top_k'] = top_answers
                if pred_i['answer'] not in ['yes', 'no', 'irrelevant']:
                    pred_i['answer'] = top_ans
                if CLASSES[ex['feat']['answer_class']] == 'more':
                    pred_i['answer_span_start'] = top_s
                    pred_i['answer_span_end'] = top_e
                    pred_i['answer_span_text'] = top_ans
        return preds

    def extract_preds_leaderboard(self, out, batch):
        preds = super().extract_preds_leaderboard(out, batch)
        scores = out['span_scores']
        ystart, yend = scores.split(1, dim=-1)
        pstart = F.softmax(ystart.squeeze(-1), dim=1)
        pend = F.softmax(yend.squeeze(-1), dim=1)

        for pred_i, pstart_i, pend_i, ex, pointer_mask_i in zip(preds, pstart, pend, batch, out['pointer_mask']):
            if pred_i['answer'] not in ['yes', 'no', 'irrelevant']:
                # start inclusive, end exclusive
                sentence_start = ex['feat']['memory_idx'].to(self.device) + 1  # +1 to skip the cls token, inclusive
                sentence_end = torch.cat([ex['feat']['memory_idx'][1:].to(self.device), (pointer_mask_i == 1).nonzero()[-1] + 1])  # exclusive
                top_preds = []
                for sent_s, sent_e in zip(sentence_start, sentence_end):
                    top_start = self.get_top_k(pstart_i, self.top_k, sent_s, sent_e)
                    top_end = self.get_top_k(pend_i, self.top_k, sent_s, sent_e)
                    for s, ps in top_start:
                        for e, pe in top_end:
                            if e >= s:
                                top_preds.append((s, e, ps * pe))
                top_preds = sorted(top_preds, key=lambda tup: tup[-1], reverse=True)[:self.top_k]
                top_answers = [(detokenize(ex['feat']['inp'][s:e + 1]), s, e, p) for s, e, p in top_preds]
                top_ans, top_s, top_e, top_p = top_answers[0]
                pred_i['answer'] = top_ans
        return preds

    def compute_f1(self, span_gold, span_pred):
        gold_toks = list(range(span_gold['s'], span_gold['e']+1))
        pred_toks = list(range(span_pred['s'], span_pred['e']+1))
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return 0
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_metrics(self, preds, data):
        metrics = super().compute_metrics(preds, data)

        span_f1 = []
        span_preds = {ex['utterance_id']: {'s': ex['answer_span_start'], 'e': ex['answer_span_end']} for ex in preds}
        span_golds = {ex['utterance_id']: {'s': ex['feat']['answer_span_start'], 'e': ex['feat']['answer_span_end']} for ex in data}
        for id in span_golds.keys():
            span_pred = span_preds[id]
            span_gold = span_golds[id]
            if span_gold['s'] != -1:
                assert span_pred['s'] != -1
                span_f1.append(self.compute_f1(span_gold, span_pred))
        metrics['span_f1'] = float("{0:.2f}".format(sum(span_f1)/len(span_f1) * 100))
        # metrics['combined'] = float("{0:.2f}".format(0.3 * metrics['macro_accuracy']/0.8 + 0.3 * metrics['micro_accuracy']/0.75 +
        #                                              0.15 * metrics['span_f1']/0.65 + 0.125 * metrics['bleu_1']/0.55 +
        #                                              0.125 * metrics['bleu_4']/0.45))
        return metrics

    def compute_loss(self, out, batch):
        loss = super().compute_loss(out, batch)

        scores = out['span_scores']
        ystart, yend = scores.split(1, dim=-1)
        gstart = torch.tensor([e['feat']['answer_span_start'] for e in batch], dtype=torch.long, device=self.device)
        loss['span_start'] = F.cross_entropy(ystart.squeeze(-1), gstart, ignore_index=-1)
        gend = torch.tensor([e['feat']['answer_span_end'] for e in batch], dtype=torch.long, device=self.device)
        loss['span_end'] = F.cross_entropy(yend.squeeze(-1), gend, ignore_index=-1)
        loss['span_start'] *= self.args.loss_span_weight
        loss['span_end'] *= self.args.loss_span_weight

        return loss
