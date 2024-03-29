import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from einops import rearrange 

class ContextMatching(nn.Module):
  #self.cmat = ContextMatching(n_dim * 3)
  def __init__(self, channel_size):
    super(ContextMatching, self).__init__()
    
  def similarity(self, s1, l1, s2, l2):
    #similarity between s1, s2 
    s = torch.bmm(s1, s2.transpose(1,2))
    
    s_mask = s.data.new(*s.size()).fill_(1).byte() #[B, T1, T2]
    
    #similarity mask using lengths
    
    s_mask = Variable(s_mask)
    s.data.masked_fill_(s_mask.data.byte(), -float("inf"))
    return s 
  
  #u_q = self.cmat(ctx, ctx_l, q_embed, q_l)
  #u_a = [self.cmat(ctx, ctx_l, a_embed[i], a_l[i] for i in range(5)]
  #we use 4 heads and 75 dimensions for d_k at multi-head attention layer
class CharMatching(nn.Module):
  #self.mhattn_script = CharMatching(4, D, D)
  def __init__(self, heads, hidden, d_model, dropout=0.1):
    super(CharMatching.self).__init__()
    self.mhatt = MHAttn(heads, hidden, d_model, dropout)
    self.ffn = FFN(d_model, d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.norm1 = Norm(d_model)
    self.dropout2 = nn.Dropout(dropout)
    self.norm2 = Norm(d_model)
    
  #u_ch = [mhattn(qa_character[i], ctx, ctx_l) for i in range(5)]  
  def forward(self, q, kv, mask_len):
    # take input stream H_S, H_M, H_B as key K to find parts directly related to the character appearing
    # in the question and answer. 
    # q_i as query q
    # we project the query and each input stream timestep respectively to h hidden projections of d_k dimesions
    # with h differnet parameter matrices 
    # then dot product attention is calculated between each input projection and query projection
    # a_i is the attention score for each timestep of input stream at i-th projection
    #W_q_i and W_k_i are i-th weight matrices
    
    # we expand attention scores and multiply to each projection vector
    # after concatenating all heads in the second dimension, a linear layer is applied to ensure that 
    # shape matches the input stream 
    # as an output of multi-head attention has the same shape as the input stream
    # add the input and output with normalization
    
    att_v = kv
    mask, _ = self.len_to_mask(mask_len, mask_len.max())
