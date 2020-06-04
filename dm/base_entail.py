import os
import shutil
import torch
import logging
import importlib
import numpy as np
import json
from tqdm import trange
from pprint import pformat
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from preprocess_dm import detokenize, compute_metrics, CLASSES
from pytorch_pretrained_bert import BertModel, BertAdam
from argparse import Namespace
from statistics import stdev, mean


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class Module(nn.Module):

    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.epoch = 0
        bert_model = args.bert_model_path
        self.bert = BertModel.from_pretrained(bert_model, cache_dir=None)
        self.dropout = nn.Dropout(self.args.dropout)
        # Memory Cell
        self.w_value = nn.Linear(self.args.bert_hidden_size, self.args.bert_hidden_size, bias=True)
        self.w_key = nn.Linear(self.args.bert_hidden_size, self.args.bert_hidden_size, bias=True)
        self.w_input = nn.Linear(self.args.bert_hidden_size, self.args.bert_hidden_size, bias=True)
        # Output
        self.w_selfattn = nn.Linear(self.args.bert_hidden_size * 2, 1, bias=True)
        self.w_output = nn.Linear(self.args.bert_hidden_size * 2, 4, bias=True)
        self.w_entail = nn.Linear(self.args.bert_hidden_size * 2, 3, bias=True)

    @classmethod
    def load_module(cls, name):
        return importlib.import_module('dm.{}'.format(name)).Module

    @classmethod
    def load(cls, fname, override_args=None):
        load = torch.load(fname, map_location=lambda storage, loc: storage)
        args = vars(load['args'])
        if override_args:
            args.update(override_args)
        args = Namespace(**args)
        model = cls.load_module(args.model)(args)
        model.load_state_dict(load['state'])
        return model

    def save(self, metrics, dsave, early_stop):
        files = [os.path.join(dsave, f) for f in os.listdir(dsave) if f.endswith('.pt') and f != 'best.pt']
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if len(files) > self.args.keep-1:
            for f in files[self.args.keep-1:]:
                os.remove(f)

        fsave = os.path.join(dsave, 'step{}-{}.pt'.format(metrics['global_step'], metrics[early_stop]))
        torch.save({
            'args': self.args,
            'state': self.state_dict(),  # comment to save space
            'metrics': metrics,
        }, fsave)
        fbest = os.path.join(dsave, 'best.pt')
        if os.path.isfile(fbest):
            os.remove(fbest)
        shutil.copy(fsave, fbest)

    def create_input_tensors(self, batch):
        feat = {
            k: torch.stack([e['feat'][k] for e in batch], dim=0).to(self.device)
            for k in ['input_ids', 'type_ids', 'input_mask', 'pointer_mask']
        }
        return feat

    def forward(self, batch):
        out = self.create_input_tensors(batch)
        out['bert_enc'], _ = bert_enc, _ = self.bert(out['input_ids'], out['type_ids'], out['input_mask'], output_all_encoded_layers=False)

        # get representation for key-value memory & input, key=value=rules, input=initialQ/scenario/qa_history
        # input_len: length of input; memory_mask: [1] mask for length of memory
        key, value, input, input_len, memory_mask = [], [], [], [], []
        for idx, e in enumerate(batch):
            key.append(torch.index_select(bert_enc[idx], 0, e['feat']['memory_idx'].to(self.device)))
            value.append(torch.index_select(bert_enc[idx], 0, e['feat']['memory_idx'].to(self.device)))
            input.append(torch.index_select(bert_enc[idx], 0, e['feat']['input_idx'].to(self.device)))
            input_len.append(e['feat']['input_idx'].shape[0])
            memory_mask.append(torch.tensor([1] * e['feat']['memory_idx'].shape[0], dtype=torch.uint8))
        key_padded = torch.nn.utils.rnn.pad_sequence(key)
        value_padded = torch.nn.utils.rnn.pad_sequence(value)
        input_padded = torch.nn.utils.rnn.pad_sequence(input)

        # run sequential update
        value_padded_history = []  # trace value after every input update
        out['gate'] = []
        gate_memory_mask = torch.nn.utils.rnn.pad_sequence(memory_mask)
        for step in range(input_padded.shape[0]):
            input_current = input_padded[step,:].unsqueeze(0)
            gate = torch.sigmoid(torch.sum(input_current * key_padded, dim=-1, keepdim=True) + \
                                 torch.sum(input_current * value_padded, dim=-1, keepdim=True))  # use key & value as gate
            gate_input_mask = torch.tensor(input_len).gt(step).unsqueeze(0).repeat(gate_memory_mask.shape[0], 1)
            gate_mask = gate_memory_mask * gate_input_mask
            out['gate'].extend(torch.masked_select(gate.squeeze(-1).to('cpu'), gate_mask).tolist())
            # gate = torch.sigmoid(torch.sum(input_current * key_padded, dim=-1, keepdim=True))  # only use key as gate
            value_updated_padded = F.relu(self.w_input(self.dropout(input_current)).repeat(key_padded.shape[0], 1, 1) + \
                                          self.w_key(self.dropout(key_padded)) +\
                                          self.w_value(self.dropout(value_padded)))
            value_updated_padded = gate * value_updated_padded + value_padded
            # only calculate norm of un-padded indices
            value_updated_norm = torch.abs(torch.norm(value_updated_padded, p=2, dim=-1, keepdim=True)) + 1e-8
            value_padded = value_updated_padded / value_updated_norm
            value_padded_history.append(value_padded)

        # extract effective values
        value_effecitve = torch.stack([value_padded_history[input_len_i-1][:,bs_i,:] for bs_i, input_len_i in enumerate(input_len)], dim=1)
        key_value = torch.cat([value_effecitve, key_padded], dim=-1)  # kv-selfattn
        out['key_value'] = key_value

        # create a mask for self attention
        unmask_selfattn_weight = self.w_selfattn(self.dropout(key_value))
        # create a mask for values
        selfattn_mask = torch.nn.utils.rnn.pad_sequence(memory_mask).unsqueeze(-1).to(self.device)
        out['sentence_mask'] = selfattn_mask
        unmask_selfattn_weight.masked_fill_(~selfattn_mask, -float('inf'))
        out['identify_scores'] = unmask_selfattn_weight.squeeze(-1).transpose(0,1)
        out['alpha_scores'] = unmask_selfattn_weight.squeeze(-1).transpose(0, 1)
        masked_selfattn_weight = F.softmax(unmask_selfattn_weight, dim=0)

        # final decision
        decision_vector = torch.sum(masked_selfattn_weight * key_value, dim=0)
        out['clf_scores'] = self.w_output(self.dropout(decision_vector))

        # entailment
        out['entail_scores'] = self.w_entail(self.dropout(key_value))
        return out

    def extract_preds(self, out, batch):
        preds = []
        for idx, (ex, clf_i, rule_i) in enumerate(zip(batch, out['clf_scores'].max(1)[1].tolist(),
                                     out['identify_scores'].max(1)[1].tolist(),)):
            a = CLASSES[clf_i]
            questionworthy_sent = -1
            if a not in ['yes', 'no', 'irrelevant']:
                a = detokenize(ex['ann']['snippet_t'][rule_i])
            # here we evaluate identify acc whenever gold decision is more
            if CLASSES[ex['feat']['answer_class']] == 'more':
                questionworthy_sent = rule_i
            snippet = ' '.join([detokenize(sent_tok) for sent_tok in ex['ann']['snippet_t']])
            pred_entail = out['entail_scores'][:,idx,:].max(1)[1][:ex['feat']['sent_state'].shape[0]].tolist()
            preds.append({
                'utterance_id': ex['utterance_id'],
                'answer': a,
                'questionworthy_sent': questionworthy_sent,
                'snippet': snippet,
                'entail': pred_entail,
            })
        return preds

    def extract_preds_leaderboard(self, out, batch):
        preds = []
        for idx, (ex, clf_i, rule_i) in enumerate(zip(batch, out['clf_scores'].max(1)[1].tolist(),
                                     out['identify_scores'].max(1)[1].tolist(),)):
            a = CLASSES[clf_i]
            if a not in ['yes', 'no', 'irrelevant']:
                a = detokenize(ex['ann']['snippet_t'][rule_i])
            snippet = ' '.join([detokenize(sent_tok) for sent_tok in ex['ann']['snippet_t']])
            preds.append({
                'utterance_id': ex['utterance_id'],
                'answer': a,
                'snippet': snippet,
            })
        return preds

    def compute_loss(self, out, batch):
        gclf = torch.tensor([ex['feat']['answer_class'] for ex in batch], device=self.device, dtype=torch.long)
        # calculate loss for entailment
        gentail = torch.nn.utils.rnn.pad_sequence([ex['feat']['sent_state'] for ex in batch], padding_value=-1).to(self.device).view(-1)
        loss = {
            'clf': F.cross_entropy(out['clf_scores'], gclf),
            'entail': F.cross_entropy(out['entail_scores'].view(-1, 3), gentail, ignore_index=-1) * self.args.loss_entail_weight,
        }
        return loss

    def compute_metrics(self, preds, data):
        from sklearn.metrics import accuracy_score, confusion_matrix
        # preds = [{'utterance_id': p['utterance_id'], 'answer': p['top_k'][0][0]} for p in preds]
        metrics = compute_metrics(preds, data)
        metrics['combined'] = float("{0:.2f}".format(metrics['macro_accuracy'] * metrics['micro_accuracy'] / 100))
        # entailment
        entail_preds = [i for ex in preds for i in ex['entail']]
        entail_golds = [i for ex in data for i in ex['feat']['sent_state'].tolist()]
        entail_micro_accuracy = accuracy_score(entail_golds, entail_preds)
        metrics["entail_micro_accuracy"] = float("{0:.2f}".format(entail_micro_accuracy * 100))  # int(100 * micro_accuracy) / 100
        entail_conf_mat = confusion_matrix(entail_golds, entail_preds, labels=[0,1,2])
        entail_conf_mat_norm = entail_conf_mat.astype('float') / entail_conf_mat.sum(axis=1)[:, np.newaxis]
        entail_macro_accuracy = np.mean([entail_conf_mat_norm[i][i] for i in range(entail_conf_mat.shape[0])])
        metrics["entail_macro_accuracy"] = float("{0:.2f}".format(entail_macro_accuracy * 100))  # int(100 * macro_accuracy) / 100
        return metrics

    def run_pred(self, dev):
        preds = []
        self.eval()
        gates = []
        for i in range(0, len(dev), self.args.dev_batch):
            batch = dev[i:i+self.args.dev_batch]
            out = self(batch)
            gates.extend(out['gate'])
            preds += self.extract_preds(out, batch)
        return preds, gates

    def run_pred_leaderboard(self, dev):
        preds = []
        self.eval()
        for i in range(0, len(dev), self.args.dev_batch):
            batch = dev[i:i+self.args.dev_batch]
            out = self(batch)
            preds += self.extract_preds_leaderboard(out, batch)
        return preds

    def run_train(self, train, dev):
        if not os.path.isdir(self.args.dsave):
            os.makedirs(self.args.dsave)

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.args.dsave, 'train.log'))
        fh.setLevel(logging.CRITICAL)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        logger.addHandler(ch)

        num_train_steps = int(len(train) / self.args.train_batch * self.args.epoch)

        # remove pooler
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        optimizer = BertAdam(optimizer_grouped_parameters, lr=self.args.learning_rate, warmup=self.args.warmup, t_total=num_train_steps)

        print('num_train', len(train))
        print('num_dev', len(dev))

        global_step = 0
        best_metrics = {self.args.early_stop: -float('inf')}
        for epoch in trange(self.args.epoch, desc='epoch',):
            self.epoch = epoch
            train = train[:]
            np.random.shuffle(train)

            train_stats = defaultdict(list)
            gates = []
            preds = []
            self.train()
            for i in trange(0, len(train), self.args.train_batch, desc='batch'):
                actual_train_batch = int(self.args.train_batch / self.args.gradient_accumulation_steps)
                batch_stats = defaultdict(list)
                batch = train[i: i + self.args.train_batch]

                for accu_i in range(0, len(batch), actual_train_batch):
                    actual_batch = batch[accu_i : accu_i + actual_train_batch]
                    out = self(actual_batch)
                    gates.extend(out['gate'])
                    pred = self.extract_preds(out, actual_batch)
                    loss = self.compute_loss(out, actual_batch)

                    for k, v in loss.items():
                        loss[k] = v / self.args.gradient_accumulation_steps
                        batch_stats[k].append(v.item()/ self.args.gradient_accumulation_steps)
                    sum(loss.values()).backward()
                    preds += pred

                lr_this_step = self.args.learning_rate * warmup_linear(global_step/num_train_steps, self.args.warmup)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                for k in batch_stats.keys():
                    train_stats['loss_' + k].append(sum(batch_stats[k]))

                if global_step % self.args.eval_every_steps == 0:
                    dev_stats = defaultdict(list)
                    dev_preds, dev_gates = self.run_pred(dev)
                    dev_metrics = {k: sum(v) / len(v) for k, v in dev_stats.items()}
                    dev_metrics.update(self.compute_metrics(dev_preds, dev))
                    dev_metrics.update({'gate_avg': mean(dev_gates)})
                    dev_metrics.update({'gate_std': stdev(dev_gates)})
                    metrics = {'global_step': global_step}
                    # metrics.update({'train_' + k: v for k, v in train_metrics.items()})
                    metrics.update({'dev_' + k: v for k, v in dev_metrics.items()})
                    logger.critical(pformat(metrics))

                    if metrics[self.args.early_stop] > best_metrics[self.args.early_stop]:
                        logger.critical('Found new best! Saving to ' + self.args.dsave)
                        best_metrics = metrics
                        self.save(best_metrics, self.args.dsave, self.args.early_stop)
                        with open(os.path.join(self.args.dsave, 'dev.preds.json'), 'wt') as f:
                            json.dump(dev_preds, f, indent=2)
                        with open(os.path.join(self.args.dsave, 'dev.best_metrics.json'), 'wt') as f:
                            json.dump(best_metrics, f, indent=2)

                    self.train()

            train_metrics = {k: sum(v) / len(v) for k, v in train_stats.items()}
            train_metrics.update(self.compute_metrics(preds, train))
            train_metrics.update({'gate_avg': mean(gates)})
            train_metrics.update({'gate_std': stdev(gates)})

            dev_stats = defaultdict(list)
            dev_preds, dev_gates = self.run_pred(dev)
            dev_metrics = {k: sum(v) / len(v) for k, v in dev_stats.items()}
            dev_metrics.update(self.compute_metrics(dev_preds, dev))
            dev_metrics.update({'gate_avg': mean(dev_gates)})
            dev_metrics.update({'gate_std': stdev(dev_gates)})
            metrics = {'global_step': global_step}
            metrics.update({'train_' + k: v for k, v in train_metrics.items()})
            metrics.update({'dev_' + k: v for k, v in dev_metrics.items()})
            logger.critical(pformat(metrics))

            if metrics[self.args.early_stop] > best_metrics[self.args.early_stop]:
                logger.critical('Found new best! Saving to ' + self.args.dsave)
                best_metrics = metrics
                self.save(best_metrics, self.args.dsave, self.args.early_stop)
                with open(os.path.join(self.args.dsave, 'dev.preds.json'), 'wt') as f:
                    json.dump(dev_preds, f, indent=2)
                with open(os.path.join(self.args.dsave, 'dev.best_metrics.json'), 'wt') as f:
                    json.dump(best_metrics, f, indent=2)

        logger.critical('Best dev')
        logger.critical(pformat(best_metrics))
