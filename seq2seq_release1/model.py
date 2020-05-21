import math
import torch
import random
from torch import nn
import torchsnooper
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class LSTMGait(nn.Module):
    def __init__(self,batch_size,hidden_size):
        super(LSTMGait, self).__init__()
        self.hidden_size =hidden_size
        self.w_omega=Variable(torch.zeros(256*2,128).cuda()).to(torch.float64)
        self.u_omega=Variable(torch.zeros(128).cuda()).to(torch.float64)   ###128,attention_size
        self.FC=nn.Linear(in_features=36, out_features=72)
        self.lstm = nn.LSTM(input_size=72, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.attention =Attention(hidden_size)
        self.classfier = nn.Linear(in_features=self.hidden_size*2, out_features=124)
        # self.bn = nn.BatchNorm1d(num_features=24,affine=False)


    # # @torchsnooper.snoop()
    def attention_net(self,lstm_output):
        b,s,n=lstm_output.shape   #[batch_size,seq_len,num_direction*hiddensize]
        output_shape =torch.Tensor.reshape(lstm_output,[-1,n])   #[batch_size*seq_len,num_direction*hiddensize]
        attn_tanh =torch.tanh(torch.mm(output_shape,self.w_omega))   #[batch_size*seq_len,attention_size] 矩阵相乘（线性代数中矩阵相乘），激活函数
        attn_hidden_layer =torch.mm(attn_tanh,torch.Tensor.reshape(self.u_omega, [-1,1]))  #[batch_size*seq_len,1]
        exps =torch.Tensor.reshape(torch.exp(attn_hidden_layer),[-1,s])  #[batch_size,seq_len]
        alphas =exps/torch.Tensor.reshape(torch.sum(exps,1),[-1,1])      #[batch_size,seq_len]
        alphas_reshape =torch.reshape(alphas,[-1,s,1])                  #[batch_size,seq_len,1]
        state=lstm_output                                                #[batch_size,seq_len,num_direnctions*hidden_size]
        attn_output =torch.sum(state*alphas_reshape,1)                  #[batch_size,num_direnctions*hidden_size]
        return attn_output

    # @torchsnooper.snoop()    #####装饰函数,print variable tensor of shape\ dtype\grd\device id
    def forward(self, input):
        # self.lstm.flatten_parameters()
        b,s,c =input.size()
        input =self.FC(input.to(torch.float32)).double()
        # input =self.bn(input.to(torch.float32)).double() ######the train loss could not improve just keep
        output, (h_n, c_n) = self.lstm(input)  ####out_put[32,24,512],h_n[2,32,512],c_n[2,32,512]

        ##############version 4: output  process and attention net###############
        attn_output=self.attention_net(output)
        y=self.classfier(attn_output)

        ##############version 3: output  process and attention net###############
        # encoder_outputs = (output[:, :, :self.hidden_size] +
        #            output[:, :, self.hidden_size:])
        # last_hidden =h_n[:1]
        # attn_weights = self.attention(last_hidden , encoder_outputs)   ######atten_weights [32,1,24]
        # print(attn_weights)       ######暂时搁置这边的论文想法

        ##############version 1: output  process###############
        # output = output.permute(0, 2, 1)
        # ########直接对最后一层的所有time-step输出output，进行平均池化####
        # f = F.avg_pool1d(output,s )       ###f [32,256,1]
        # f = f.view(b, self.hidden_size)   ###f [32,256]
        # y = self.classfier(f)

        #####version 2:直接对每一层的最后time-step的输出h，进行池化操作#########
        # output_in_last_timesstep =h_n[-1,:,:]
        # y = self.classfier(output_in_last_timesstep)
        return y,attn_output


#######swj model#######
# Model2 batch_first=True
class RNN(torch.nn.Module):
    def __init__(self, input_size, channel_size, step_size, embed_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size=input_size
        self.step_size =step_size
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.output_size=output_size

        self.fc1 = torch.nn.Linear(input_size * channel_size, embed_size)  # (feature size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.lstm = torch.nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True,
                                  bidirectional=True)  # batch_first=True , num_layers=2, dropout=0.9
        self.fc2 = torch.nn.Linear(hidden_size * 2, output_size)  # hidden_size*2 #0002

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_size * 2,1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.cpu().numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    # @torchsnooper.snoop()
    def forward(self, x):
        # input为(batch_size,seq_len,input_szie)
        h0 = Variable(torch.randn(1 * 2, x.size(0), self.hidden_size).cuda()).to(torch.float64)
        c0 = Variable(torch.randn(1 * 2, x.size(0), self.hidden_size).cuda()).to(torch.float64)

        # Method 1
        B, S, F = x.size()  # Batch, Step, Input, Chanel
        x = x.view(B * S, F)
        embed = self.fc1(x)
        embed = self.bn(embed)
        embed = embed.view(B, S, self.embed_size).to(torch.float64)  # (-1, step_size, embed_size) embed_size=128

        states, (hn, cn) = self.lstm(embed, (h0, c0))  # embed size(seq_len, batch, iput_size) (states, hidden)
        # states, hidden = self.lstm(embed)

        # original
        #         #encoding = torch.cat([states[0], states[-1]], dim=1)#0001 第二个维度相加， ->batch, 2*hidden_size
        #         encoding = states[:, -1, :] #0002
        #         outputs = self.fc2(encoding)

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        attn_output, attention = self.attention_net(states, hn)

        # return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]
        outputs = self.fc2(attn_output)

        return outputs, attn_output


#######language transform based model######
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # timestep = encoder_outputs.size(0)
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)   ##沿着0维度拷贝timestep次，沿着1维度拷贝1次，沿着2维度拷贝1次，然后0和1维度互换
        # encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2).to(torch.float32)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()      ####max_len stands for time step

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs
