import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from einops import rearrange 

class ContextMatching(nn.Module):
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
  
class CharMatching(nn.Module):
  def __init__(self, heads, hidden, d_model, dropout=0.1):
    super(CharMatching.self).__init__()
    self.mhatt = MHAttn(heads, hidden, d_model, dropout)
    self.ffn = FFN(d_model, d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.norm1 = Norm(d_model)
    self.dropout2 = nn.Dropout(dropout)
    self.norm2 = Norm(d_model)
    
    
  def forward(self, q, kv, mask_len):
    att_v = kv
    mask, _ = self.len_to_mask(mask_len, mask_len.max())
