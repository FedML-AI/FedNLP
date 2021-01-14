import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

VERY_BIG_NUMBER = 1e30
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

class BIDAF_SpanExtraction(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, char_emb_len, word_emb_len, out_channel_dims, filter_heights,
    max_word_len, char_out_size, dropout_rate, hidden_size, highway_num_layers):
        super(BIDAF_SpanExtraction, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.char_emb_len = char_emb_len
        self.word_emb_len = word_emb_len
        self.out_channel_dims = list(map(int, out_channel_dims.split(',')))
        self.filter_heights = list(map(int, filter_heights.split(',')))
        self.max_word_len = max_word_len
        self.char_out_size = char_out_size
        self.dropout_rate = dropout_rate
        self.highway_num_layers = highway_num_layers
        self.hidden_size = hidden_size

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(self.char_vocab_size, self.char_emb_len)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv_layers = nn.ModuleList()
        for out_channel_dim, filter_height in zip(self.out_channel_dims, self.filter_heights):
            self.char_conv_layers.append(nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Conv2d(1, out_channel_dim, (self.char_emb_len, filter_height)),
                nn.ReLU()
                ))

        # word embedding layer
        self.word_emb = nn.Parameter(torch.Tensor(self.word_vocab_size, self.word_emb_len), requires_grad=True)
        nn.init.normal_(self.word_emb)

        # highway network
        assert self.hidden_size * 2 == (sum(self.out_channel_dims) + self.word_emb_len)
        for i in range(self.highway_num_layers):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = nn.LSTM(input_size=self.hidden_size * 2,
                                 hidden_size=self.hidden_size,
                                 bidirectional=True,
                                 batch_first=True)

        # 4. Attention Flow Layer
        self.att_weight_c = nn.Linear(self.hidden_size * 2, 1)
        self.att_weight_q = nn.Linear(self.hidden_size * 2, 1)
        self.att_weight_cq = nn.Linear(self.hidden_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = nn.LSTM(input_size=self.hidden_size * 8,
                                   hidden_size=self.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)

        self.modeling_LSTM2 = nn.LSTM(input_size=self.hidden_size * 2,
                                   hidden_size=self.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)

        # 6. Output Layer
        self.p1_weight_g = nn.Linear(self.hidden_size * 8, 1)
        self.p1_weight_m = nn.Linear(self.hidden_size * 2, 1)
        self.p2_weight_g = nn.Linear(self.hidden_size * 8, 1)
        self.p2_weight_m = nn.Linear(self.hidden_size * 2, 1)

        self.output_LSTM = nn.LSTM(input_size=self.hidden_size * 14,
                                hidden_size=self.hidden_size,
                                bidirectional=True,
                                batch_first=True)

        self.dropout = nn.Dropout(p=self.dropout_rate)
        

    def forward(self, x, cx, x_mask, q, cq, q_mask, y, y2, glove_emb_weights, device):
        """
        x -> [batch_size, max_sent_len]
        cx -> [batch_size, max_sent_len, word_len]
        q -> [batch_size, max_ques_len]
        cq -> [batch_size, max_ques_len, word_len]
        y -> [batch_size, max_sent_len]
        y2 -> [batch_size, max_sent_len]
        x_mask -> [batch_size, max_sent_len]
        q_mask -> [batch_size, max_ques_len]
        """


        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            seq_len = x.size(1)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch， seq_len, char_dim, word_len)
            x = x.transpose(2, 3)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.char_emb_len, x.size(3)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            out = []
            for char_conv_layers in self.char_conv_layers:
                out.append(char_conv_layers(x).squeeze())
            x = torch.cat(out, dim=-1)
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, seq_len, -1)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(self.highway_num_layers):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.dropout(self.p1_weight_g(g)) + self.dropout(self.p1_weight_m(m))).squeeze()
            # (batch, c_len, hidden_size * 2)
            m1 = torch.sum(torch.softmax(torch.unsqueeze(p1, -1), -1) * m, dim=1, keepdim=True).repeat(1, p1.size(1), 1)
            # (batch, c_len, hidden_size * 2)
            # m2 = self.dropout(self.output_LSTM(m)[0])
            m2 = self.dropout(self.output_LSTM(torch.cat([g, m, m1, m * m1], -1))[0])
            # (batch, c_len)
            p2 = (self.dropout(self.p2_weight_g(g)) + self.dropout(self.p2_weight_m(m2))).squeeze()

            return p1, p2

        word_emb = torch.cat([self.word_emb, glove_emb_weights], 0)

        # 1. Character Embedding Layer
        c_char = char_emb_layer(cx)
        q_char = char_emb_layer(cq)
        # 2. Word Embedding Layer
        c_word = F.embedding(x, word_emb)
        q_word = F.embedding(q, word_emb)
        c_lens = torch.sum(x_mask.int(), dim=1)
        q_lens = torch.sum(q_mask.int(), dim=1)

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)
        # 3. Contextual Embedding Layer
        c = self.dropout(self.context_LSTM((c))[0])
        q = self.dropout(self.context_LSTM((q))[0])
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 5. Modeling Layer
        m = self.dropout(self.modeling_LSTM2(self.dropout(self.modeling_LSTM1(g)[0]))[0])
        # 6. Output Layer
        p1, p2 = output_layer(g, m, c_lens)

        # (batch, c_len), (batch, c_len)
        return p1, p2

        




