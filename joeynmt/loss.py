# coding: utf-8
"""
Module to implement custom loss functions
"""

import torch
from torch import nn, Tensor
from torch.autograd import Variable


class LabelSmoothingLoss(nn.Module):
    """
    Cross-Entropy Loss with label smoothing
    """

    def __init__(self, ignore_index: int, smoothing: float = 0.1,
                 reduction: str = "sum"):
        super(LabelSmoothingLoss, self).__init__()
        assert smoothing > 0, "Use nn.CrossEntropyLoss for the unsmoothed case"
        self.smoothing = smoothing
        self.pad_index = ignore_index
        # custom label-smoothed loss, computed with KL divergence loss
        self.criterion = nn.KLDivLoss(reduction=reduction)

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0-self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, input, targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param input: logits as predicted by model
        :param targets: target indices
        :return:
        """
        log_probs = torch.log_softmax(input, dim=-1)

        targets = self._smooth_targets(
            targets=targets.contiguous().view(-1),
            vocab_size=log_probs.size(-1))

        # targets: distributions with batch*seq_len x vocab_size
        assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape \
            == targets.shape
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        return loss
