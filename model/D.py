import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import math
from . rnn import RNNEncoder, max_along_time, Embedding
from . modules import CharMatching, ContextMatching
from . layers import SelfAtt, BiDAFOutput, Encoder

def _to_mask(lengths, len_max):
  #print(len(len_max))
  mask = torch.arange(len_max, device=lengths.device, dtype=lengths.dtype).expand(len(lengths), len_max) >=lengths.unsqueeze(1)
  #mask = torch.zeros_like(mask, device = lengths.device, dtype=lengths.dtype) != mask
  mask = torch.as_tensor(mask, dtype=torch.bool, device=lengths.device)
  print("_to_mask mask: ", mask.size())
  return mask
#mask [batch_size, c_len]
def new_parameter(*size):
  out = nn.Parameter(torch.cuda.FloatTensor(*size))
  nn.init.xavier_normal(out)
  return out

class HighwayEncoder(nn.Module):
  def __init__(self, num_layers, hidden_size):
    super(HighwayEncoder, self).__init__()
    self.trans = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
    self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

  def forward(self, input):
    for gate, transform in zip(self.gates, self.trans):
      g = gate(input)
      g = torch.sigmoid(g)
      t = transform(input)
      t = F.relu(t)
      output = g*t+(1-g)*input
      return output


class D(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()
        print("Model name: Dual Matching Multistream")
        self.vocab = vocab
        V = len(vocab)
        D = n_dim

        self.embedding = nn.Embedding(V, D)
        n_dim = args.n_dim
        image_dim = args.image_dim


        self.cmat = ContextMatching(n_dim * 3) 
        self.lstm_raw = RNNEncoder(300, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        self.lstm_script = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        self.script_on = "script" in args.stream_type
        self.vbb_on = "visual_bb" in args.stream_type
        self.vmeta_on = "visual_meta" in args.stream_type
        self.conv_pool = Conv1d(n_dim*4+1, n_dim*2)

        self.character = nn.Parameter(torch.randn(22, D, device=args.device, dtype=torch.float), requires_grad=True)
        self.norm1 = Norm(D)
        self.emb_qa = nn.Linear(300, 300)
        self.emb_o = nn.Linear(321, 321)
        self.self_att = SelfAtt(hidden_size=n_dim, drop_prob=0.)
        self.mod = Encoder(input_size=n_dim, hidden_size=n_dim, num_layers = 2, drop_prob = 0.)
        self.out = BiDAFOutput(hidden_size=n_dim, drop_prob=0.)

        if self.script_on:
            self.lstm_script = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
            self.lstm_mature_script = RNNEncoder(n_dim  * 5, n_dim, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_script = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))
            self.mhattn_script = CharMatching(4, D, D)

        if self.vmeta_on:            
            self.lstm_vmeta = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

            self.lstm_mature_vmeta = RNNEncoder(n_dim  * 5, n_dim, bidirectional=True,
                                               dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vmeta = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))
            self.mhattn_vmeta = CharMatching(4, D, D)

        if self.vbb_on:
            self.lstm_vbb = RNNEncoder(image_dim+21, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

            self.vbb_fc = nn.Sequential(
               nn.Dropout(0.5),
               nn.Linear(image_dim, n_dim),
               nn.Tanh(),
               #nn.Linear(image_dim, image_dim, bias=False),
               #nn.ReLU(),
               #nn.Linear(image_dim, n_dim, bias=False),
            )
            self.lstm_mature_vbb = RNNEncoder(n_dim * 2 * 5, n_dim, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vbb = nn.Sequential(nn.Linear(n_dim*2, 1), nn.Softmax(dim=1))

            self.mhattn_vbb = CharMatching(4, D, D)



    def _to_one_hot(self, y, n_dims, mask, dtype=torch.cuda.FloatTensor):
        scatter_dim = len(y.size())
        y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), n_dims).type(dtype)
        out = zeros.scatter(scatter_dim, y_tensor, 1)

        out_mask,_ = self.len_to_mask(mask, out.shape[1])
        out_mask = out_mask.unsqueeze(2).repeat(1, 1, n_dims)

        return out.masked_fill_(out_mask, 0)


    def load_embedding(self, pretrained_embedding):
        print('Load pretrained embedding ...')
        #self.embedding.weight.data.copy_(pretrained_embedding)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def len_to_mask(self, lengths, len_max):
        #len_max = lengths.max().item()
        mask = torch.arange(len_max, device=lengths.device,
                        dtype=lengths.dtype).expand(len(lengths), len_max) >= lengths.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=torch.uint8, device=lengths.device)

        return mask, len_max

    def forward(self, que, answers, **features):
        '''
        filtered_sub (B, max_sub_len)
        filtered_sub_len (B)
        filtered_speaker (B, max_sub_len)
        filtered_visual (B, max_v_len*3)
        filtered_visual_len (B)
        filtered_image (B, max_v_len, 512)
        filtered_image_len (12)
        que (B, max_que_len)
        que_len (B)
        answers (B, 5, max_ans_len)
        ans_len (B, 5)
        
        print(que.shape)
        print(answers.shape)
        for key, value in features.items():
            print(key, value.shape)
            
        '''
        #batch size is 16
       
        q_level_logic = features['q_level_logic']
       
        print("q_level_logic: ", q_level_logic)
        '''
        x_1 = 0
        x_2 = 0
        x_3 = 0
        x_4 = 0
        for i in range(16):
          x = q_level_logic[i]
          if x == 1:
            x_1+=1
          elif x == 2:
            x_2+=1
          elif x == 3:
            x_3+=1
          else:
            x_4+=1
        print("q_level_logic: ", x_1, x_2, x_3, x_4)
        '''
        B = que.shape[0]
        # -------------------------------- #
        e_q = self.embedding(que)
        e_q = self.emb_qa(e_q)
        q_len = features['que_len']
        e_q, _ = self.lstm_raw(e_q, q_len)

        # -------------------------------- #
        e_ans = self.embedding(answers).transpose(0, 1)
        e_ans = self.emb_qa(e_ans)
        ans_len = features['ans_len'].transpose(0, 1)
        e_ans_list = [self.lstm_raw(e_a, ans_len[idx])[0] for idx, e_a in enumerate(e_ans)]


        concat_qa = [(self.get_name(que, q_len) + self.get_name(answers.transpose(0,1)[i], ans_len[i])).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [(torch.sum(concat_qa[i], dim=1) == 0).unsqueeze(1).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [torch.cat([concat_qa[i], concat_qa_none[i]], dim=1) for i in range(5)]
        q_c = [torch.matmul(concat_qa_none[i], self.character) for i in range(5)]
        q_c = [self.norm1(q_c[i]) for i in range(5)]

        if self.script_on:
            e_s = self.embedding(features['filtered_sub'])
            s_len = features['filtered_sub_len']
            # -------------------------------- #
            spk = features['filtered_speaker']
            spk_onehot = self._to_one_hot(spk, 21, mask=s_len)
            e_s = torch.cat([e_s, spk_onehot], dim=2)
            e_s = self.emb_o(e_s)
            spk_flag = [torch.matmul(spk_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            spk_flag = [(spk_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            H_S, _ = self.lstm_script(e_s, s_len)
            o_s = self.stream_processor(self.classifier_script,self.mhattn_script, spk_flag, H_S, s_len, q_c, e_q, q_len, e_ans_list, ans_len)
        else:
            o_s = 0

        if self.vmeta_on:
            vmeta = features['filtered_visual'].view(B, -1, 3)
            vmeta_len = features['filtered_visual_len']*2/3
             
            vp = vmeta[:,:,0]
            vp = vp.unsqueeze(2).repeat(1,1,2).view(B, -1)
            vbe = vmeta[:,:,1:3].contiguous()
            vbe = vbe.view(B, -1)
            e_vbe = self.embedding(vbe)
            # -------------------------------- #
            vp_onehot = self._to_one_hot(vp, 21, mask=vmeta_len)
            e_vbe = torch.cat([e_vbe, vp_onehot], dim=2)
            e_vbe = self.emb_o(e_vbe)
            vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            H_M, _ = self.lstm_vmeta(e_vbe, vmeta_len)
            o_m = self.stream_processor(self.classifier_vmeta, self.mhattn_vmeta, vp_flag, H_M, vmeta_len, q_c, e_q, q_len, e_ans_list, ans_len)
        else:
            o_m = 0

        if self.vbb_on:
            e_vbb = features['filtered_person_full']
            vbb_len = features['filtered_person_full_len']
            
            #fimage = features['filtered_image']
            #vmeta_len = features['filtered_image_len']

            vp = features['filtered_visual'].view(B, -1, 3)[:,:,0]
            vp = vp.unsqueeze(2).view(B, -1)
            # -------------------------------- #
            vp_onehot = self._to_one_hot(vp, 21, mask=vbb_len)
            e_vbb =self.vbb_fc(e_vbb)
            e_vbb = torch.cat([e_vbb, vp_onehot], dim=2)
            e_vbb = self.emb_o(e_vbb)
            vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            # -------------------------------- #
            H_B, _ = self.lstm_script(e_vbb, vbb_len)
            o_b = self.stream_processor(self.classifier_vbb, self.mhattn_vbb, vp_flag, H_B, vbb_len, q_c, e_q, q_len, e_ans_list, ans_len)

        else:
            o_b = 0

        #out = (x_3)*o_s + (x_3)*o_m + (x_1+x_2)*o_b
        out = o_s + o_m + o_b
        #model selects the answer candidate with the largest final output score 
        return out.view(B, -1)

       
    def stream_processor(self, classifier, mhattn, ctx_flag, ctx, ctx_l,
                         qa_character, q_embed, q_l, a_embed, a_l):
        # boolean flag f which is true when the speaker or the person in visual metadata appears
        # in the question and answer pair
        u_q = self.cmat(ctx, ctx_l, q_embed, q_l)
        #print("u_q: ", u_q.shape)
        u_a = [self.cmat(ctx, ctx_l, a_embed[i], a_l[i]) for i in range(5)]
        #print("u_a: ", [u_a[i].shape for i in range(5)])
        u_ch = [mhattn(qa_character[i], ctx, ctx_l) for i in range(5)]
        #print("u_ch:", u_ch.shape)
        #print("I want to  print 7: ", u_q[1].size()[0])
        #print("u_a[0][1].size()[0]: ", u_a[0][1].size()[0])
        #q_mask = _to_mask(ctx_l, u_q[1].size()[0])
        #a_mask = [_to_mask(ctx_l, u_a[i][1].size()[0]) for i in  range(5)]
        
        #u_q = self.self_att(u_q, q_mask)
        #print("after self att  u_q: ", u_q.shape)
        #u_a = [self.self_att(u_a[i], a_mask[i]) for i in range(5)]
        #print("after self att u_a: ", [u_a[i].shape for i in range(5)])
        #m_q = self.mod(u_q, ctx_l)
        #print("m_q: ", m_q.shape)
        #m_a = [self.mod(u_a[i], ctx_l) for i in range(5)]
        #log_q1, log_q2 = self.out(u_q, m_q, q_mask)[0]
        #log_a1  = [self.out(u_a[i], m_a[i], a_mask[i])[0] for i in range(5)]
        #log_a2 = [self.out(u_a[i], m_a[i], a_mask[i])[1] for i in range(5)]
       # concat_a = [torch.cat([ctx, u_ch[i],u_q, u_a[i],  ctx_flag[i]], dim=-1) for i in range(5)]
        concat_a = [torch.cat([ctx, u_q, u_a[i], u_ch[i], ctx_flag[i]], dim=-1) for i in range(5)] 
        #concat_a = [torch.cat([ctx, u_q, u_a[i], ctx_flag[i], u_ch[i]],dim=-1) for i in range(5)]
        # ctx, u_ch[i], ctx_flag[i],
        # exp_2 : ctx, u_a[i], u_q, ctx_flag[i], u_ch[i]
        maxout = [self.conv_pool(concat_a[i], ctx_l) for i in range(5)]

        answers = torch.stack(maxout, dim=1)
        out = classifier(answers)  # (B, 5)

        return out 

    def get_name(self, x, x_l):
        x_mask = x.masked_fill(x>20, 21)
        x_onehot = self._to_one_hot(x_mask, 22, x_l)
        x_sum = torch.sum(x_onehot[:,:,:21], dim=1)
        return x_sum > 0




class Conv1d(nn.Module):
    def __init__(self, n_dim, out_dim):
        super().__init__()
        # apply 1-D convolution filters with various kernel sizes and concatenate them to get 
        # final representation
        # applying max-pool over time and linear layer 
        # we calculate scalar score for each candidate answer 
        out_dim = int(out_dim/4)
        self.conv_k1 = nn.Conv1d(n_dim, out_dim, kernel_size=1, stride=1)
        self.conv_k2 = nn.Conv1d(n_dim, out_dim, kernel_size=2, stride=1)
        self.conv_k3 = nn.Conv1d(n_dim, out_dim, kernel_size=3, stride=1)
        self.conv_k4 = nn.Conv1d(n_dim, out_dim, kernel_size=4, stride=1)
       # self.maxpool = nn.MaxPool1d(kernel_size = 1)

    def forward(self, x, x_l):
        # x : (B, T, 5*D)
        x_pad = torch.zeros(x.shape[0],3,x.shape[2]).type(torch.cuda.FloatTensor)
        x = torch.cat([x, x_pad], dim=1)
        x1 = F.relu(self.conv_k1(x.transpose(1,2)))[:,:,:-3]
        x2 = F.relu(self.conv_k2(x.transpose(1,2)))[:,:,:-2]
        x3 = F.relu(self.conv_k3(x.transpose(1,2)))[:,:,:-1]
        x4 = F.relu(self.conv_k4(x.transpose(1,2)))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        #out = self.maxpool(out)
        out = out.transpose(1,2)
        return max_along_time(out, x_l)


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
          # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
