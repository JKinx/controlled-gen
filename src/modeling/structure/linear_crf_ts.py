import torch
import copy

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from .. import torch_model_utils as tmu
from ..torch_struct import LinearChainCRF as LCRF

class LinearChainCRF(nn.Module):
  def __init__(self, config):
    super(LinearChainCRF, self).__init__()
    self.label_size = config.latent_vocab_size

    init_transition = torch.randn(
      self.label_size, self.label_size).to(config.device)

    # do not use any start or end index, assume that all links to start and end
    # has potential 1 
    self.transition = nn.Parameter(init_transition)
    return 

  def calculate_all_scores(self, emission_scores):
    """Mix the transition and emission scores

    Args:
      emission_scores: type=torch.Tensor(float), 
        size=[batch, max_len, num_class]

    Returns:
      scores: size=[batch, len, num_class, num_class]
      scores = log phi(batch, x_t, y_{t-1}, y_t)
    """
    label_size = self.label_size
    batch_size = emission_scores.size(0)
    seq_len = emission_scores.size(1)

    # scores[batch, t, C, C] = log_potential(t, from y_{t-1}, to y_t)
    scores = self.transition.view(1, 1, label_size, label_size)\
      .expand(batch_size, seq_len, label_size, label_size) + \
      emission_scores.view(batch_size, seq_len, 1, label_size)\
      .expand(batch_size, seq_len, label_size, label_size)

    return scores

  def marginals(self, emission_scores, seq_lens):
    all_scores = self.calculate_all_scores(emission_scores)
    dist = LCRF(all_scores.transpose(3,2), (seq_lens + 1).float())
    return dist.marginals.sum(-1)

  def rsample(self, emission_scores, seq_lens, tau):
    all_scores = self.calculate_all_scores(emission_scores)
    dist = LCRF(all_scores.transpose(3,2), (seq_lens + 1).float())
    return dist.gumbel_crf(tau).sum(-1)

  def entropy(self, emission_scores, seq_lens):
    all_scores = self.calculate_all_scores(emission_scores)
    dist = LCRF(all_scores.transpose(3,2), (seq_lens + 1).float())
    return dist.entropy