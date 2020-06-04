# use entailment score to mudulate span score for span extraction
import torch
from dm.span_entail import Module as Base
from torch import nn
from torch.nn import functional as F


class Module(Base):
    def forward(self, batch):
        out = super().forward(batch)
        # TODO actually here is a bug, we need to mask padded sentences first before we do sft
        entail_score = F.softmax(out['entail_scores'], dim=-1)[:, :, 0].transpose(0, 1)  # sft version
        # entail_score = out['entail_scores'][:, :, 0].transpose(0, 1)
        # use entail_scores to modulate span_scores
        sentence_score = torch.ones(out['span_scores'].shape, dtype=torch.float, device=self.device)
        for batch_idx, (unknown_entail_score_i, ex, pointer_mask_i) in enumerate(zip(entail_score, batch, out['pointer_mask'])):
            sentence_start = ex['feat']['memory_idx'].to(self.device) + 1  # +1 to skip the cls token, inclusive
            sentence_end = torch.cat([ex['feat']['memory_idx'][1:].to(self.device), (pointer_mask_i == 1).nonzero()[-1] + 1])  # exclusive
            for sent_idx, (s, e) in enumerate(zip(sentence_start, sentence_end)):
                sentence_score[batch_idx,s:e,:] = unknown_entail_score_i[sent_idx]
        out['span_scores'] *= sentence_score
        return out

