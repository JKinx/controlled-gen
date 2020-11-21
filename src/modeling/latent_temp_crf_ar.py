"""Latent template CRF, autoregressive version"""

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Uniform

from .lstm_seq2seq.encoder import LSTMEncoder
from .lstm_seq2seq.decoder import LSTMDecoder, Attention
from .structure.linear_crf import LinearChainCRF
from . import torch_model_utils as tmu
import operator

from torch_struct import LinearChainCRF as LC

class LatentTemplateCRFAR(nn.Module):
  """The latent template CRF autoregressive version, table to text setting"""

  def __init__(self, config, embeddings=None):
    super().__init__()
    self.config = config
    self.device = config.device

    self.z_beta = config.z_beta
    self.z_overlap_logits = config.z_overlap_logits
    self.z_sample_method = config.z_sample_method
    self.gumbel_st = config.gumbel_st
    self.use_src_info = config.use_src_info
    self.use_copy = config.use_copy
    self.num_sample = config.num_sample

    self.num_sample_rl = config.num_sample_rl
    self.z_gamma = config.z_gamma # switching loss
    self.z_b0 = config.z_b0 # constant baseline
    self.z_lambd = config.z_lambd # reward scaling
    self.reward_level = config.reward_level

    self.pad_id = config.pad_id
    self.start_id = config.start_id
    self.end_id = config.end_id
    self.seg_id = config.seg_id

    self.vocab_size = config.vocab_size
    self.latent_vocab_size = config.latent_vocab_size 

    self.lstm_layers = config.lstm_layers
    self.embedding_size = config.embedding_size
    self.state_size = config.state_size
    self.max_dec_len = config.max_dec_len
    self.max_bow_len = config.max_bow_len

    self.z_pred_strategy = config.z_pred_strategy
    self.x_pred_strategy = config.x_pred_strategy

    ## Model parameters
    # emb
    self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
    if(embeddings is not None): 
      self.embeddings.weight.data.copy_(torch.from_numpy(embeddings))
    self.z_embeddings = nn.Embedding(
      config.latent_vocab_size, config.embedding_size)
    # enc
    self.q_encoder = LSTMEncoder(config)
    # latent 
    self.z_crf_proj = nn.Linear(config.state_size, config.latent_vocab_size)
    self.z_crf = LinearChainCRF(config)
    # dec 
    self.p_dec_init_state_proj_h = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.p_dec_init_state_proj_c = nn.Linear(
      config.embedding_size, config.lstm_layers * config.state_size)
    self.p_decoder = LSTMDecoder(config)
    # copy 
    self.p_copy_attn = Attention(
      config.state_size, config.state_size, config.state_size)
    self.p_copy_g = nn.Linear(config.state_size, 1)
    # dec z proj
    self.p_z_proj = nn.Linear(config.state_size, config.latent_vocab_size)
    self.p_z_intermediate = nn.Linear(2 * config.state_size, config.state_size)
    
    # posterior regularization
    self.pr = config.pr
    self.pr_lambd = config.pr_lambd
    self.num_pr_constraints = config.num_pr_constraints
    
    return 

  def init_state(self, s):
    batch_size = s.shape[0]
    init_state_h = self.p_dec_init_state_proj_h(s)
    init_state_h = init_state_h.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_h = init_state_h.transpose(0, 1).contiguous()
    init_state_c = self.p_dec_init_state_proj_c(s)
    init_state_c = init_state_c.view(
      batch_size, self.lstm_layers, self.state_size)
    init_state_c = init_state_c.transpose(0, 1).contiguous()
    return (init_state_h, init_state_c)

  def encode_kv(self, keys, vals):
    """Encode the key-valud table"""
    kv_mask = keys != self.pad_id 
    keys_emb = self.embeddings(keys)
    vals_emb = self.embeddings(vals)
    kv_emb = keys_emb + vals_emb # [batch, mem_len, state_size]

    kv_mask_ = kv_mask.type(torch.float)
    kv_enc = kv_emb * kv_mask_.unsqueeze(-1)
    # kv_enc.shape = [batch, embedding_size]
    kv_enc = kv_enc.sum(dim=1) / kv_mask_.sum(dim=1, keepdim=True)
    return kv_emb, kv_enc, kv_mask

  def forward(self, keys, vals, 
    sentences, sent_lens, tau, x_lambd, return_grad=False, zcs=None):
    """Forward pass, first run the inference network, then run the decoder
    
    Args:
      keys: torch.tensor(torch.long), size=[batch, max_mem_len]
      vals: torch.tensor(torch.long), size=[batch, max_mem_len]
      sentences: torch.tensor(torch.long), size=[batch, sent_len]
      sent_lens: torch.tensor(torch.long), size=[batch]
      tau: gumbel temperature, anneal from 1 to 0.01
      x_lambd: decoder coefficient for the word in, controll how 'autogressive'
       the model is, anneal from 0 to 1 

    Returns:
      loss: torch.float, the total loss 
      out_dict: dict(), output dict  
      out_dict['inspect']: dict(), training process inspection
    """
    out_dict = {}
    inspect = {}
    batch_size = sentences.size(0)
    device = sentences.device
    loss = 0.

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    # enc_outputs.shape = [batch, max_len, state_size]
    enc_outputs, (enc_state_h, enc_state_c) =\
      self.q_encoder(sentences_emb, sent_lens)
    # NOTE: max_len != sentences.size(1), max_len = max(sent_lens)
    max_len = enc_outputs.size(1)
    sent_mask = sent_mask[:, : max_len]

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    ## latent template
    # emission score = log potential
    # [batch, max_len, latent_vocab]
    z_emission_scores = self.z_crf_proj(enc_outputs) 
    if(self.z_overlap_logits):
      z_emission_scores[:, :-1] += z_emission_scores[:, 1:].clone()
      z_emission_scores[:, 1:] += z_emission_scores[:, :-1].clone()
        
    # PR
    if self.pr:
      out_dict["sent_mask"] = sent_mask
      out_dict["z_scores"] = z_emission_scores
        
      # get pr loss
      pr_loss = self.compute_pr(z_emission_scores, zcs[:, :max_len], sent_mask, sent_lens)
      out_dict["pr_val"] = tmu.to_np(pr_loss)
      out_dict["pr_loss"] = self.pr_lambd * tmu.to_np(pr_loss)
      loss -= self.pr_lambd * pr_loss
        
    
    # entropy regularization
    ent_z = self.z_crf.entropy(z_emission_scores, sent_lens).mean()
    loss += self.z_beta * ent_z
    out_dict['ent_z'] = tmu.to_np(ent_z)
    out_dict['ent_z_loss'] = self.z_beta * tmu.to_np(ent_z)

    # reparameterized sampling
    if(self.z_sample_method == 'gumbel_ffbs'):
      z_sample_ids, z_sample, _ = self.z_crf.rsample(
        z_emission_scores, sent_lens, tau, return_switching=True)
    elif(self.z_sample_method == 'pm'):
      z_sample_ids, z_sample = self.z_crf.pmsample(
        z_emission_scores, sent_lens, tau)
    else:
      raise NotImplementedError(
        'z_sample_method %s not implemented!' % self.z_sample_method)

    z_sample_max, _ = z_sample.max(dim=-1)
    z_sample_max = z_sample_max.masked_fill(~sent_mask, 0)
    inspect['z_sample_max'] = (z_sample_max.sum() / sent_mask.sum()).item()
    out_dict['z_sample_max'] = inspect['z_sample_max']

    # NOTE: although we use 0 as mask here, 0 is ALSO a valid state 
    z_sample_ids.masked_fill_(~sent_mask, 0) 
    z_sample_ids_out = z_sample_ids.masked_fill(~sent_mask, -1)
    out_dict['z_sample_ids'] = tmu.to_np(z_sample_ids_out)
    inspect['z_sample_ids'] = tmu.to_np(z_sample_ids_out)
    z_sample_emb = tmu.seq_gumbel_encode(z_sample, z_sample_ids,
      self.z_embeddings, self.gumbel_st)

    # decoding
    sentences = sentences[:, : max_len]
    p_log_prob, _, p_log_prob_x, p_log_prob_z, z_acc, _ = self.decode_train(
      z_sample_ids, z_sample_emb, sent_lens,
      keys, kv_emb, kv_enc, kv_mask, sentences, x_lambd)
    out_dict['p_log_prob'] = p_log_prob.item()
    out_dict['p_log_prob_x'] = p_log_prob_x.item()
    out_dict['p_log_prob_z'] = p_log_prob_z.item()
    out_dict['z_acc'] = z_acc.item()
    loss += p_log_prob

    # # turn maximization to minimization
    loss = -loss

    if(return_grad):
      self.zero_grad()
      g = torch.autograd.grad(
        loss, z_emission_scores, retain_graph=True)[0]
      g_mean = g.mean(0)
      g_std = g.std(0)
      g_r =\
        g_std.log() - g_mean.abs().log()
      out_dict['g_mean'] =\
        g_mean.abs().log().mean().item()
      out_dict['g_std'] = g_std.log().mean().item()
      out_dict['g_r'] = g_r.mean().item()   

    out_dict['loss'] = tmu.to_np(loss)
    out_dict['inspect'] = inspect
    return loss, out_dict

  def compute_pr(self, emission_scores, zcs, sent_mask, sent_lens):
    # edge potentials
    all_scores = self.z_crf.calculate_all_scores(emission_scores)
    
    # Linear Chain CRF
    dist = LC(all_scores.transpose(3,2), (sent_lens + 1).float())
    
    # marginals : [batch, max_len, state_size]
    marginals = dist.marginals.sum(-1)
    rel_marginals = marginals[:, :, :self.num_pr_constraints]
    
    # filters for the constrained z states
    filters = torch.arange(self.num_pr_constraints).view(1,1,self.num_pr_constraints).to(self.device)
    
    # check if state is used
    check = zcs.unsqueeze(-1) == filters
    
    # computer loss
    # {1 - q(z = sigma(f) | x, y)} if f is used else {q(z = sigma(f) | x, y)}
    loss = (check.float() - rel_marginals).abs() * sent_mask.unsqueeze(-1).float()
    
    # take average
    final_loss = loss.sum() / sent_lens.sum()
    
    return final_loss
  
  def forward_score_func(self, keys, vals, 
    sentences, sent_lens, x_lambd, num_sample, return_grad=False):
    out_dict = {}
    inspect = {}
    batch_size = sentences.size(0)
    device = sentences.device
    loss = 0.

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    # enc_outputs.shape = [batch, max_len, state_size]
    enc_outputs, (enc_state_h, enc_state_c) =\
      self.q_encoder(sentences_emb, sent_lens)
    # NOTE: max_len != sentences.size(1), max_len = max(sent_lens)
    max_len = enc_outputs.size(1)
    sent_mask = sent_mask[:, : max_len]

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    ## latent template
    # emission score = log potential
    # [batch, max_len, latent_vocab]
    z_emission_scores = self.z_crf_proj(enc_outputs) 
    if(self.z_overlap_logits):
      z_emission_scores[:, :-1] += z_emission_scores[:, 1:].clone()
      z_emission_scores[:, 1:] += z_emission_scores[:, :-1].clone()

    # entropy regularization
    ent_z = self.z_crf.entropy(z_emission_scores, sent_lens).mean()
    loss += self.z_beta * ent_z
    out_dict['ent_z'] = ent_z.item()
    out_dict['ent_z_loss'] = self.z_beta * ent_z.item()

    # reparameterized sampling
    z_emission_scores = tmu.batch_repeat(z_emission_scores, num_sample)
    sent_lens = tmu.batch_repeat(sent_lens, num_sample)
    z_sample_ids, z_sample, _, z_log_prob, z_transition = self.z_crf.rsample(
      z_emission_scores, sent_lens, 
      tau=0.01, return_switching=True, return_prob=True)
    z_transition = z_transition.view(batch_size, num_sample, -1)

    sent_mask = tmu.batch_repeat(sent_mask, num_sample)
    out_dict['z_sample_ids'] = tmu.to_np(
      z_sample_ids.view(batch_size, num_sample, -1)[:, 0])
    z_sample_max, _ = z_sample.max(dim=-1)
    z_sample_max = z_sample_max.masked_fill(~sent_mask, 0)
    inspect['z_sample_max'] = (z_sample_max.sum() / sent_mask.sum()).item()
    out_dict['z_sample_max'] = inspect['z_sample_max']

    # NOTE: although we use 0 as mask here, 0 is ALSO a valid state 
    z_sample_ids.masked_fill_(~sent_mask, 0) 
    z_sample_ids_out = z_sample_ids.masked_fill(~sent_mask, -1)\
      .view(batch_size, num_sample, -1)
    inspect['z_sample_ids'] = tmu.to_np(z_sample_ids_out[:, 0])
    z_sample_emb = tmu.seq_gumbel_encode(z_sample, z_sample_ids,
      self.z_embeddings, gumbel_st=True)

    # decoding
    sentences = sentences[:, : max_len]
    sentences = tmu.batch_repeat(sentences, num_sample)
    z_sample_ids = z_sample_ids.detach()
    z_sample_emb = z_sample_emb.detach()
    keys = tmu.batch_repeat(keys, num_sample)
    kv_emb = tmu.batch_repeat(kv_emb, num_sample)
    kv_enc = tmu.batch_repeat(kv_enc, num_sample)
    kv_mask = tmu.batch_repeat(kv_mask, num_sample)
    (p_log_prob, p_log_prob_casewise, p_log_prob_x, p_log_prob_z, 
      z_acc, p_log_prob_stepwise) = self.decode_train(
        z_sample_ids, z_sample_emb, sent_lens,
        keys, kv_emb, kv_enc, kv_mask, sentences, x_lambd)
    out_dict['p_log_prob'] = p_log_prob.item()
    out_dict['p_log_prob_x'] = p_log_prob_x.item()
    out_dict['p_log_prob_z'] = p_log_prob_z.item()
    out_dict['z_acc'] = z_acc.item()
    loss += p_log_prob

    # score function estimator
    if(self.reward_level == 'seq'):
      # sequence level reward
      p_log_prob_casewise = p_log_prob_casewise.view(batch_size, num_sample)
      b = p_log_prob_casewise.detach()
      b = (b.sum(dim=1, keepdim=True) - b) / (num_sample - 1)
      z_log_prob = z_log_prob.view(batch_size, num_sample)
      reward_seq = (p_log_prob_casewise - b - self.z_b0).mean().detach()
      out_dict['reward'] = reward_seq.item()
      learning_signal_seq =\
        (p_log_prob_casewise - b - self.z_b0).detach() * z_log_prob
      learning_signal = self.z_lambd * learning_signal_seq.mean() 
      out_dict['learning_signal'] = learning_signal.item()
    elif(self.reward_level == 'step'):
      max_len = max_len - 1
      # Stepwise reward, unbiased transition version 
      p_log_prob_stepcum = p_log_prob_stepwise.view(batch_size, num_sample, 1, -1)
      p_log_prob_stepcum = p_log_prob_stepcum.repeat(1, 1, max_len, 1)
      cum_mask = torch.triu(torch.ones(max_len, max_len)).to(device)
      sent_mask_ = sent_mask.view(
        batch_size, num_sample, max_len + 1, 1)[:,:,:-1,:]
      cum_mask = cum_mask.view(1, 1, max_len, max_len) * sent_mask_.float()
      p_log_prob_stepcum = (p_log_prob_stepcum * cum_mask).sum(-1)
      # NOTE: does this baseline make sense? 
      b = p_log_prob_stepcum.detach()
      b = (b.sum(dim=1, keepdim=True) - b) / (num_sample - 1)
      learning_signal_step_ut =\
        (p_log_prob_stepcum - b - self.z_b0).detach() * z_transition[:,:,:-1]
      reward_step_ut = (p_log_prob_stepcum - b - self.z_b0).mean().detach()
      out_dict['reward'] = (reward_step_ut + 1e-20).item()
      learning_signal_step_ut = self.z_lambd * learning_signal_step_ut.mean()
      out_dict['learning_signal'] = learning_signal_step_ut.item()
      learning_signal = learning_signal_step_ut
    else: 
      raise NotImplementedError(
        'reward level %s not implemented' % self.reward_level)
      
    loss += learning_signal

    # turn maximization to minimization
    loss = -loss

    if(return_grad):
      self.zero_grad()
      g = torch.autograd.grad(
        loss, z_emission_scores, retain_graph=True)[0]
      g_mean = g.mean(0)
      g_std = g.std(0)
      g_r =\
        g_std.log() - g_mean.abs().log()
      out_dict['g_mean'] =\
        g_mean.abs().log().mean().item()
      out_dict['g_std'] = g_std.log().mean().item()
      out_dict['g_r'] = g_r.mean().item()   

    out_dict['loss'] = tmu.to_np(loss)
    out_dict['inspect'] = inspect
    return loss, out_dict

  def forward_lm(self, sentences, sent_lens):
    out_dict = {}

    # sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    dec_cell = self.p_decoder
    dec_inputs = sentences_emb[:, :-1].transpose(1, 0)
    dec_targets = sentences[:, 1:].transpose(1, 0)
    max_len = sentences.size(1) - 1

    state = self.init_state(sentences_emb.mean(1))
    state = [(state[0] * 0).detach(), (state[1] * 0).detach()]
    dec_cell = self.p_decoder
    log_prob_x = []
    for i in range(max_len):
      dec_out, state = dec_cell(dec_inputs[i], state)
      dec_out = dec_out[0]
      x_logits = dec_cell.output_proj(dec_out)
      log_prob_x_i = -F.cross_entropy(
        x_logits, dec_targets[i], reduction='none')
      log_prob_x.append(log_prob_x_i)

    log_prob_x = torch.stack(log_prob_x).transpose(1, 0) # [max_len, batch]
    log_prob_x = tmu.mask_by_length(log_prob_x, sent_lens)
    nll = log_prob_x.sum(1).mean()
    loss = -log_prob_x.sum() / sent_lens.sum()
    out_dict['loss'] = loss.item()
    out_dict['ppl'] = loss.exp().item()
    out_dict['marginal'] = nll.item()
    return loss, out_dict

  def infer_marginal(self, keys, vals, 
    sentences, sent_lens, num_sample):
    """Marginal probability and ELBO 
    Via importance sampling from the inference network
    """
    out_dict = {}
    inspect = {}
    batch_size = sentences.size(0)
    device = sentences.device
    loss = 0.

    ## sentence encoding 
    sent_mask = sentences != self.pad_id
    sentences_emb = self.embeddings(sentences)
    # enc_outputs.shape = [batch, max_len, state_size]
    enc_outputs, (enc_state_h, enc_state_c) =\
      self.q_encoder(sentences_emb, sent_lens)
    # NOTE: max_len != sentences.size(1), max_len = max(sent_lens)
    max_len = enc_outputs.size(1)
    sent_mask = sent_mask[:, : max_len]

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    ## latent template
    # emission score = log potential
    # [batch, max_len, latent_vocab]
    z_emission_scores = self.z_crf_proj(enc_outputs) 
    if(self.z_overlap_logits):
      z_emission_scores[:, :-1] += z_emission_scores[:, 1:].clone()
      z_emission_scores[:, 1:] += z_emission_scores[:, :-1].clone()

    # entropy regularization
    ent_z = self.z_crf.entropy(z_emission_scores, sent_lens).mean()
    out_dict['ent_z'] = ent_z.item()

    # reparameterized sampling
    z_emission_scores = tmu.batch_repeat(z_emission_scores, num_sample)
    sent_lens = tmu.batch_repeat(sent_lens, num_sample)
    z_sample_ids, z_sample, z_sample_log_prob, _ = self.z_crf.rsample(
      z_emission_scores, sent_lens, tau=0.01,
      return_switching=False, return_prob=True)
    z_sample_emb = tmu.seq_gumbel_encode(z_sample, z_sample_ids, 
      self.z_embeddings, gumbel_st=True)

    # decoding
    sentences = sentences[:, : max_len]
    sentences = tmu.batch_repeat(sentences, num_sample)
    keys = tmu.batch_repeat(keys, num_sample)
    kv_emb = tmu.batch_repeat(kv_emb, num_sample)
    kv_enc = tmu.batch_repeat(kv_enc, num_sample)
    kv_mask = tmu.batch_repeat(kv_mask, num_sample)
    p_log_prob, p_log_prob_casewise, p_log_prob_x, p_log_prob_z, z_acc, _ =\
      self.decode_train(
        z_sample_ids, z_sample_emb, sent_lens,
        keys, kv_emb, kv_enc, kv_mask, sentences, x_lambd=0)
    out_dict['p_log_prob_x'] = p_log_prob_x.item()
    out_dict['p_log_prob_z'] = p_log_prob_z.item()
    out_dict['z_acc'] = z_acc.item()
    
    # elbo 
    elbo = (p_log_prob_casewise - z_sample_log_prob).mean()
    out_dict['elbo'] = elbo.item()

    # marginal prob
    p_log_prob_casewise = p_log_prob_casewise.view(batch_size, num_sample)
    out_dict['p_log_prob'] = p_log_prob_casewise.mean().item()
    z_sample_log_prob = z_sample_log_prob.view(batch_size, num_sample)
    out_dict['z_sample_log_prob'] = z_sample_log_prob.mean().item()
    marginal = torch.logsumexp(p_log_prob_casewise - z_sample_log_prob, 1)
    marginal = marginal - np.log(num_sample)
    sent_lens = sent_lens.view(batch_size, num_sample)[:, 0].view(batch_size)
    ppl = (-marginal / sent_lens.float()).mean().exp()
    out_dict['ppl'] = ppl.item()
    out_dict['marginal'] = marginal.mean().item()
    return out_dict


  def prepare_dec_io(self, 
    z_sample_ids, z_sample_emb, sentences, x_lambd):
    """Prepare the decoder output g based on the inferred z from the CRF 

    Args:
      x_lambd: word dropout ratio. 1 = all dropped

    Returns:
      dec_inputs
      dec_targets_x
      dec_targets_z
    """
    batch_size = sentences.size(0)
    max_len = sentences.size(1)
    device = sentences.device

    sent_emb = self.embeddings(sentences)
    z_sample_emb[:, 0] *= 0. # mask out z[0]

    # word dropout ratio = x_lambd. 0 = no dropout, 1 = all drop out
    m = Uniform(0., 1.)
    mask = m.sample([batch_size, max_len]).to(device)
    mask = (mask > x_lambd).float().unsqueeze(2)

    dec_inputs = z_sample_emb + sent_emb * mask
    dec_inputs = dec_inputs[:, :-1]

    dec_targets_x = sentences[:, 1:]
    dec_targets_z = z_sample_ids[:, 1:]
    return dec_inputs, dec_targets_x, dec_targets_z

  def decode_train(self, 
    z_sample_ids, z_sample_emb, sent_lens,
    mem, mem_emb, mem_enc, mem_mask, sentences, x_lambd):
    """Train the decoder/ generative model. Same as 
    Li and Rush 20. Posterior Control of Blackbox Generation

    Args:

    Returns:
    """
    inspect = {}

    device = z_sample_ids.device
    state_size = self.state_size
    batch_size = sentences.size(0)

    dec_inputs, dec_targets_x, dec_targets_z = self.prepare_dec_io(
      z_sample_ids, z_sample_emb, sentences, x_lambd)
    max_len = dec_inputs.size(1)

    dec_cell = self.p_decoder
    if(self.use_src_info):
      state = self.init_state(mem_enc)
    else: 
      state = self.init_state(mem_enc)
      state = [(state[0] * 0).detach(), (state[1] * 0).detach()]

    dec_inputs = dec_inputs.transpose(1, 0) # [T, B, S]
    dec_targets_x = dec_targets_x.transpose(1, 0) # [T, B]
    dec_targets_z = dec_targets_z.transpose(1, 0)
    z_sample_emb = z_sample_emb[:, 1:].transpose(1, 0) # start from z[1]

    log_prob_x, log_prob_z, dec_outputs, z_pred = [], [], [], []

    for i in range(max_len): 
      if(self.use_src_info):
        dec_out, state = dec_cell(
          dec_inputs[i], state, mem_emb, mem_mask)
      else: 
        dec_out, state = dec_cell(
          dec_inputs[i], state)
      dec_out = dec_out[0]

      # predict z 
      z_logits = self.p_z_proj(dec_out)
      z_pred.append(z_logits.argmax(dim=-1))
      log_prob_z_i = -F.cross_entropy(
        z_logits, dec_targets_z[i], reduction='none')
      log_prob_z.append(log_prob_z_i)

      # predict x based on z 
      dec_intermediate = self.p_z_intermediate(
        torch.cat([dec_out, z_sample_emb[i]], dim=1))
      x_logits = dec_cell.output_proj(dec_intermediate)
      lm_prob = F.softmax(x_logits, dim=-1)

      if(self.use_copy): 
        _, copy_dist = self.p_copy_attn(dec_intermediate, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        x_logits = (out_prob + 1e-10).log()

      log_prob_x_i = -F.cross_entropy(
        x_logits, dec_targets_x[i], reduction='none')
      log_prob_x.append(log_prob_x_i)
      
      dec_outputs.append(x_logits.argmax(dim=-1))

    # loss
    log_prob_x = torch.stack(log_prob_x).transpose(1, 0) # [B, T]
    log_prob_x = tmu.mask_by_length(log_prob_x, sent_lens)
    log_prob_z = torch.stack(log_prob_z).transpose(1, 0)
    log_prob_z = tmu.mask_by_length(log_prob_z, sent_lens)
    log_prob_step = log_prob_x + log_prob_z # stepwise reward

    log_prob_x_casewise = log_prob_x.sum(1)
    log_prob_x = log_prob_x.sum() / sent_lens.sum()
    log_prob_z_casewise = log_prob_z.sum(1)
    log_prob_z = log_prob_z.sum() / sent_lens.sum()
    log_prob_casewise = log_prob_x_casewise + log_prob_z_casewise
    log_prob = log_prob_x + log_prob_z

    # acc 
    z_pred = torch.stack(z_pred).transpose(1, 0) # [B, T]
    dec_targets_z = dec_targets_z.transpose(1, 0)
    z_positive = tmu.mask_by_length(z_pred == dec_targets_z, sent_lens).sum() 
    z_acc = z_positive / sent_lens.sum()
    
    return (
      log_prob, log_prob_casewise, log_prob_x, log_prob_z, z_acc, log_prob_step)

  def infer(self, keys, vals):
    """Latent template inference step

    Args:
      keys: size=[batch, mem_len]
      vals: size=[batch, mem_len]
      z: size=[batch, num_sample, max_len]
      z_lens: size=[batch, num_sample]

    Returns:
      out_dict
    """
    out_dict = {}
    batch_size = keys.size(0)
    mem_len = keys.size(1)
    state_size = self.state_size
    device = keys.device

    # kv encoding 
    kv_emb, kv_enc, kv_mask = self.encode_kv(keys, vals)

    # decoding 
    predictions_x, predictions_z = self.decode_infer(vals, kv_emb, kv_enc, kv_mask)
    out_dict['predictions'] = tmu.to_np(predictions_x)
    out_dict['predictions_z'] = tmu.to_np(predictions_z)
    pred_lens_ = tmu.seq_ends(predictions_x, self.end_id) + 1
    out_dict['pred_lens'] = tmu.to_np(pred_lens_)
    return out_dict

  def decode_infer(self, mem, mem_emb, mem_enc, mem_mask):
    """Inference

    Args:
      mem: torch.Tensor(), size=[batch, mem_len]
      mem_emb: torch.Tensor(), size=[batch, mem_len, state_size]
      mem_enc: torch.Tensor(), size=[batch, state_size]
      mem_mask: torch.Tensor(), size=[batch, mem_len]
    
    Returns:
      predictions_x: torch.Tensor(int), size=[batch, max_dec_len]
      predictions_z: torch.Tensor(int), size=[batch, max_dec_len]
    """
    
    batch_size = mem.size(0)
    device = mem.device

    dec_cell = self.p_decoder

    predictions_x, predictions_z = [], []
    inp = self.embeddings(
      torch.zeros(batch_size).to(device).long() + self.start_id)
    # assume use_src_info=True
    state = self.init_state(mem_enc)
    for i in range(self.max_dec_len): 
      # assume use_src_info=True
      dec_out, state = dec_cell(inp, state, mem_emb, mem_mask)
      dec_out = dec_out[0]

      # predict z 
      z_logits = self.p_z_proj(dec_out)
      if(self.z_pred_strategy == 'greedy'):
        z = z_logits.argmax(-1)
      elif(self.z_pred_strategy == 'sampling'):
        pass # TBC
      else: raise NotImplementedError(
        'Error z decode strategy %s' % self.z_pred_strategy)

      # predict x based on z 
      z_emb = self.z_embeddings(z)
      dec_intermediate = self.p_z_intermediate(
        torch.cat([dec_out, z_emb], dim=1))
      x_logits = dec_cell.output_proj(dec_intermediate)
      lm_prob = F.softmax(x_logits, dim=-1)

      if(self.use_copy):
        _, copy_dist = self.p_copy_attn(dec_intermediate, mem_emb, mem_mask)
        copy_prob = tmu.batch_index_put(copy_dist, mem, self.vocab_size)
        copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

        out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
        x_logits = (out_prob + 1e-10).log()

      if(self.x_pred_strategy == 'greedy'):
        x = x_logits.argmax(-1)
      elif(self.x_pred_strategy == 'sampling'):
        pass # TBC
      else: raise NotImplementedError(
        'Error x decode strategy %s' % self.x_pred_strategy)

      inp = z_emb + self.embeddings(x)

      predictions_x.append(x)
      predictions_z.append(z)

    predictions_x = torch.stack(predictions_x).transpose(1, 0)
    predictions_z = torch.stack(predictions_z).transpose(1, 0)
    return predictions_x, predictions_z

  def init_fstates(self, template):
    fstates = []
    plus_starts = []
    
    count = 0
    for i in range(len(template)):
      if template[i] == "(":
        assert template[i+1] != ")"
        plus_starts.append(len(fstates))
        continue
      
      if template[i] == ")":
        assert len(plus_starts) > 0
        start_id = plus_starts.pop()
        fstates[start_id]["prev_fstates"].append(len(fstates) - 1)
        continue
          
      fstate = {}
      fstate["fixed"] = template[i] != -1
      fstate["state_vocab"] = template[i]
      fstate["exit"] = i == (len(template) - 1)

      fstate["prev_fstates"] = []
      if not fstate["fixed"]:
        fstate["prev_fstates"].append(count)
      if count != 0:
        fstate["prev_fstates"].append(count-1)

      fstate["nodes"] = []
      fstates.append(fstate)
      count += 1
        
    assert plus_starts == []
    
    return fstates

  def decode_infer2(self, mem, mem_emb, mem_enc, mem_mask, templates):
    """Inference

    Args:
      mem: torch.Tensor(), size=[batch, mem_len]
      mem_emb: torch.Tensor(), size=[batch, mem_len, state_size]
      mem_enc: torch.Tensor(), size=[batch, state_size]
      mem_mask: torch.Tensor(), size=[batch, mem_len]
    
    Returns:
      predictions_x: torch.Tensor(int), size=[batch, max_dec_len]
      predictions_z: torch.Tensor(int), size=[batch, max_dec_len]
    """
    
    batch_size = mem.size(0)
    device = mem.device
    
    # beam vars
    beam_width = 2
    beam_w_ = 2
    topk = 5
    
    decoded_batch = []
    decoded_score = []
    decoded_states = []
    decoded_copy = []
    max_len = self.max_dec_len
    window_size = 3
    

    dec_cell = self.p_decoder

    predictions_x, predictions_z = [], []
    inp = self.embeddings(
      torch.zeros(batch_size).to(device).long() + self.start_id)
    
    # assume use_src_info=True
    state = self.init_state(mem_enc)
    
    
    beams_batch = []
    
    for idx in range(batch_size):
      endnodes = []
      number_required = min((topk + 1), topk - len(endnodes))
        
      fstates = self.init_fstates(templates[idx])
      beams = []
      for _ in range(len(fstates)):
        beams.append({"prev_fs":[], "prev_ks":[], "next_zs":[],"next_ys":[], 
                      "score_zs":[], "score_ys":[]})
    
      inp_i = inp[idx:idx + 1]
      state_i = (state[0][:, idx:idx + 1, :].contiguous(), state[1][:, idx:idx + 1, :].contiguous())
      mem_emb_i = mem_emb[idx:idx+1]
      mem_mask_i = mem_mask[idx:idx+1]
      mem_i = mem[idx:idx+1]
    
      t = 0
      break_flag = False
      finished = False
        
      while not finished:
#         if t>= max_len: break
        t += 1
        if len(endnodes) >= number_required: break
            
        for fstate_idx in range(len(fstates)):
          prev_nodes = []
          prev_fstate_count = 0
          
          if t == 1 and fstate_idx == 0:
            prev_fstate_count += 1
            node = {'h': state_i, 'prevNode': None, 'wordid': self.start_id, 'logp': 0, 
                    'leng': 1, 'state_id': -1, 'inp': inp_i, "fstate_idx" : fstate_idx, "idx" : -1}
            prev_nodes.append((-node['logp'], node))
            
          nextnodes = []
        
          for prev_fstates in fstates[fstate_idx]["prev_fstates"]:
            prev_fstate_count += 1
            prev_nodes += fstates[prev_fstates]["nodes"]
          prev_nodes = sorted(prev_nodes, key=lambda x: x[0])
        
          prev_fs = []
          prev_ks = [] 
          next_zs = []
          next_ys = []
          score_zs = []
          score_ys = []
        
          count = 0 
            
          for elem in range(min(len(prev_nodes), beam_width * prev_fstate_count)):
            score_top, n_top = prev_nodes[elem]
            prev_fs.append(n_top["fstate_idx"])
            prev_ks.append(n_top["idx"])
            
            if n_top["wordid"] == self.end_id and n_top["prevNode"] != None:
              endnodes.append((score, n_top))
            
              if len(endnodes) >= number_required:
                break
              else:
                continue
            
            dec_out, decoder_hidden = dec_cell(n_top['inp'].contiguous(), n_top['h'], 
                                         mem_emb_i, mem_mask_i)
            dec_out = dec_out[0]
            
            z_logits = self.p_z_proj(dec_out)
            
            state_logp = torch.log_softmax(z_logits, dim=-1)
            
            if fstates[fstate_idx]["fixed"]:
              state_id = fstates[fstate_idx]["state_vocab"]
              log_prob = state_logp[:,state_id:state_id+1]
              indexes = torch.tensor([[[state_id]]]).to(self.device)
            else:
              log_prob, indexes = torch.topk(state_logp, beam_w_)
            
            next_zs.append(indexes.view(-1).tolist())
            score_zs.append((n_top['logp'] + log_prob.view(-1)).tolist())
            next_y = []
            score_y = []
            
            for new_k in range(len(indexes.view(-1))):
              decoded_ts = indexes.view(-1)[new_k]
              log_ps = log_prob[0][new_k].item()
                
              z_emb = self.z_embeddings(torch.tensor([decoded_ts]).to(self.device))
              dec_intermediate = self.p_z_intermediate(torch.cat([dec_out, z_emb], dim=1))
              x_logits = dec_cell.output_proj(dec_intermediate)
              lm_prob = F.softmax(x_logits, dim=-1)
                
              if(self.use_copy):
                _, copy_dist = self.p_copy_attn(dec_intermediate, mem_emb_i, mem_mask_i)
                copy_prob = tmu.batch_index_put(copy_dist, mem_i, self.vocab_size)
                copy_g = torch.sigmoid(self.p_copy_g(dec_intermediate))

                out_prob = (1 - copy_g) * lm_prob + copy_g * copy_prob
                x_logits = (out_prob + 1e-10).log()
                
              if not fstates[fstate_idx]["exit"]:
                x_logits[:,self.end_id] = float("-Inf")
                
              temp_wordlp = torch.log_softmax(x_logits, dim=-1)
              
              log_prob_w, indexes_w = torch.topk(temp_wordlp, beam_width + window_size)
              
              temp_ww = []
              temp_pp = []  
                
              for elem1, elem2 in zip(indexes_w.cpu().view(-1), log_prob_w.cpu().view(-1)):
                temp_ww.append(elem1)
                temp_pp.append(elem2)

                if len(temp_ww) >= beam_width:
                  break
              
              ys = []
              ss = []
            
              for new_k_w in range(beam_width):
                decoded_tw = temp_ww[new_k_w].view(-1)[0]
                log_pw = temp_pp[new_k_w].item()
                
                inp_i = z_emb + self.embeddings(torch.tensor([decoded_tw]).to(self.device))
                
                node = {'h': decoder_hidden, 'prevNode': n_top, 'wordid': decoded_tw.item(),
                        'logp': n_top['logp'] + log_ps + log_pw, 'leng': n_top['leng'] + 1,
                        'state_id': decoded_ts.item(),  'inp': inp_i,
                         "idx":count, "fstate_idx" : fstate_idx}
                count += 1
                
                ys.append(decoded_tw.item())
                ss.append(n_top['logp'] + log_ps + log_pw)
                score = -node['logp']
                
                if node['wordid'] == self.end_id and node['prevNode'] != None:
                  endnodes.append((score, node))
                  continue
                if node["leng"] >= max_len and fstate_idx == len(fstates)-1:
                  endnodes.append((score, node))
                  continue
                else:
                  nextnodes.append((score, node))
                
              next_y.append(ys)
              score_y.append(ss)
                
            next_ys.append(next_y)
            score_ys.append(score_y)
            
          next_nodes = []
          for i in range(len(nextnodes)):
            score, nn_ = nextnodes[i]
            next_nodes.append((score, nn_))

          next_nodes = sorted(next_nodes, key=lambda x: x[0])
#                     print(fstate_idx, count,len(next_nodes))

          if next_nodes != [] and fstate_idx == len(fstates) - 1 and next_nodes[0][1]["wordid"] == self.end_id:
            finished = True
            print("finished")

          fstates[fstate_idx]["next_nodes"] = next_nodes

          beams[fstate_idx]["prev_fs"].append(prev_fs)
          beams[fstate_idx]["prev_ks"].append(prev_ks)
          beams[fstate_idx]["next_zs"].append(next_zs)
          beams[fstate_idx]["next_ys"].append(next_ys)
          beams[fstate_idx]["score_zs"].append(score_zs)
          beams[fstate_idx]["score_ys"].append(score_ys)
            
        for fstate_idx in range(len(fstates)):
          fstates[fstate_idx]["nodes"] = fstates[fstate_idx]["next_nodes"]
        
      beams_batch.append(beams)
      utterances_w = []
      utterances_s = []
      scores_result = []
    
      for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance_w = []
        utterance_s = []
        utterance_w.append(n['wordid'])
        utterance_s.append(n['state_id'])
        # back trace
        while n['prevNode'] != None:
          n = n['prevNode']
          utterance_w.append(n['wordid'])
          utterance_s.append(n['state_id'])

        utterance_w = utterance_w[::-1]
        utterance_s = utterance_s[::-1]
        utterances_w.append(utterance_w)
        utterances_s.append(utterance_s)
        scores_result.append(score)
        
      decoded_batch.append(utterances_w)
      decoded_states.append(utterances_s)
      decoded_score.append(scores_result)
        
    return decoded_batch, decoded_states, decoded_score
