import torch
import torch.nn as nn
import math
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ConvBlock1D(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size=3):
        super(ConvBlock1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_plane, out_plane, kernel_size,1,int(kernel_size)//2, bias=False),
            nn.BatchNorm1d(out_plane),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if not len(x.shape) == 3:
            print("the shape should be (N,C,M)")
            return None
        return self.conv(x)

    
class BasicBlock1D(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(BasicBlock1D, self).__init__()
        if in_plane != out_plane:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_plane, out_plane, 1,1,0, bias=False),
                nn.BatchNorm1d(out_plane)
            )
        else:
            self.downsample = None
        self.conv1 = nn.Conv1d(in_plane, out_plane, 3,1,1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_plane)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_plane, out_plane, 3,1,1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.bn2(self.conv2(
            self.relu1(self.bn1(self.conv1(x)))
        ))
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.relu2(out+residual)
        return out        


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
        
class CNNRNN_Attn(nn.Module):
    def __init__(self, n_filters, embedding_dim=1024, rop_prob=0.5,
                 nhidden=256, nlayers=1, bidirectional=True, nsent=2048):
        super(CNNRNN_Attn, self).__init__()
        self.bidirectional = bidirectional
        self.num_direction = 2 if bidirectional else 1
        self.embedding_dim = embedding_dim
        self.nhidden = nhidden // self.num_direction
        self.rnn_layers = nlayers
        self.nsent = nsent // self.num_direction
        self.BN1 = nn.Sequential(
            # 1 x 40 x frames
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, kernel_size=(n_filters, 1), stride=(1,1), padding=(0,0), bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.block = BasicBlock1D
        
        self.Conv = nn.Sequential(
            self.block(64, 64), # 64 x frames
            nn.MaxPool1d(2,2),
            self.block(64, 128), # 64 x 1 x frames//2
            nn.MaxPool1d(2,2),
            self.block(128, 256), # 128 x 1 x frames//4
            nn.MaxPool1d(2,2),
            self.block(256, 256), # 128 x 1 x frames//8
            nn.MaxPool1d(2,2),
            self.block(256, 512), # 256 x 1 x frames//16
            nn.MaxPool1d(2,2),
            # self.block(512, 512), # 256 x 1 x frames//32
            # nn.MaxPool1d(2,2),
            self.block(512, self.embedding_dim), # 256 x 1 x frames//64
            nn.MaxPool1d(2,2) # 1024 x 1 x frame//64
        )
        
        self.RNN = nn.LSTM(self.embedding_dim, self.nhidden, num_layers=self.rnn_layers, batch_first=True, bidirectional=self.bidirectional)
        self.out_RNN = nn.LSTM(self.nhidden*self.num_direction, self.nsent, num_layers=self.rnn_layers, batch_first=True, bidirectional=self.bidirectional)
        self.attn = MultiHeadAttention(4, self.nhidden*self.num_direction)
        # self.apply(self.weights_init)

    def init_hidden(self, x, n_dim):
        batch_size = x.shape[0]
        rtn = (torch.zeros(self.num_direction, batch_size, n_dim, device=x.device).requires_grad_(),
                torch.zeros(self.num_direction, batch_size, n_dim, device=x.device).requires_grad_())
        return rtn
        
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, cap_lens):
        # (batch, channel, 40, 2048)
        # print(x.shape, x.device, next(self.parameters()).device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        #print("before BN: shape: {}".format(x.shape))
        x = self.BN1(x).squeeze(2)
        #print("before Conv: shape: {}".format(x.shape))
        x = self.Conv(x)
        #print("after Conv: shape: {}".format(x.shape))
        x = x.transpose(1,2)
        cap_lens = (cap_lens).data.tolist()
        batch_size, length = x.shape[:2]
        h0 = self.init_hidden(x, self.nhidden)
        x = pack_padded_sequence(x, cap_lens, batch_first=True)
        output, hidden = self.RNN(x, h0)
        output = pad_packed_sequence(output, batch_first=True, total_length=length)[0]
        #print(output.shape)
        output = output.view(batch_size, length, self.rnn_layers, self.num_direction*self.nhidden)
        # out = hidden[0].view(batch_size, self.num_direction*self.nhidden)
        # print(output.shape)
        out = self.attn(output, output, output)
        # [B, length, embedding_dim]
        # print(out.shape)

        words_emb = out.transpose(1, 2)
        # [B, embedding_dim, length]

        h0 = self.init_hidden(out, self.nsent)
        # print(h0[0].device)
        out = pack_padded_sequence(out, cap_lens, batch_first=True)
        output, hidden = self.out_RNN(out, h0)
        # output = pad_packed_sequence(output, batch_first=True, total_length=length)[0]

        
        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nsent * self.num_direction)
        # print(words_emb.shape, sent_emb.shape)
        return words_emb, sent_emb
        
