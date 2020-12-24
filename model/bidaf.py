import torch
from torch import nn

VERY_BIG_NUMBER = 1e30
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

class BIDAF_SpanExtraction(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, char_emb_len, word_emb_len, out_channel_dims, filter_heights, 
    share_cnn_weights, max_word_len, char_out_size, keep_prob, share_lstm_weights, hidden_size,
    use_char_emb, 
    use_word_emb, 
    use_glove_for_unk,
    use_highway_network,
    highway_num_layers, glove_emb_weights=None):
        super(BIDAF_SpanExtraction, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.char_emb_len = char_emb_len
        self.word_emb_len = word_emb_len
        self.out_channel_dims = list(map(int, out_channel_dims.split(',')))
        self.filter_heights = list(map(int, filter_heights.split(',')))
        self.share_cnn_weights = share_cnn_weights
        self.max_word_len = max_word_len
        self.char_out_size = char_out_size
        self.keep_prob = keep_prob
        self.use_highway_network = use_highway_network
        self.use_glove_for_unk = use_glove_for_unk
        self.use_char_emb = use_char_emb
        self.use_word_emb = use_word_emb
        self.highway_num_layers = highway_num_layers
        self.share_lstm_weights = share_lstm_weights
        self.hidden_size = hidden_size
        input_size = 0
        if use_char_emb:
            self.char_emb = nn.Embedding(self.char_vocab_size, self.char_emb_len)
            input_size += self.char_out_size
        if use_word_emb:
            if self.use_glove_for_unk and glove_emb_weights is not None:
                word_emb_weights = torch.cat([torch.Tensor(self.word_vocab_size, self.word_emb_len), glove_emb_weights], dim=0)
                self.word_emb = nn.Embedding.from_pretrained(word_emb_weights, freeze=False)
            else:
                self.word_emb = nn.Embedding(self.word_vocab_size, self.word_emb_len)
            input_size += self.word_emb_len
        self.conv1d_layers = nn.ModuleList()
        for out_channel_dim, filter_height in zip(self.out_channel_dims, self.filter_heights):
            self.conv1d_layers.append(nn.Conv1d(self.char_emb_len, out_channel_dim, (1, filter_height)))
        if not share_cnn_weights:
            self.conv1d_layers_2 = nn.ModuleList()
            for out_channel_dim, filter_height in zip(self.out_channel_dims, self.filter_heights):
                self.conv1d_layers_2.append(nn.Conv1d(self.char_emb_len, out_channel_dim, (1, filter_height)))
        
        if self.use_highway_network:
            self.highway_layers = nn.ModuleList()
            for idx in range(self.highway_num_layers):
                # trans layer
                self.highway_layers.append(nn.Linear(input_size, input_size))
                # gate layer
                self.highway_layers.append(nn.Linear(input_size, input_size))
        
        self.prepro_lstm_layer = nn.LSTM(input_size, self.hidden_size, bidirectional=True)
        if not self.share_lstm_weights:
            self.prepro_lstm_layer_2 = nn.LSTM(input_size, self.hidden_size, bidirectional=True)
        
        self.biattention_layer = nn.Linear(6 * self.hidden_size, 1)

        self.main_lstm_layer = nn.LSTM(8 * self.hidden_size, self.hidden_size, bidirectional=True)
        self.main_lstm_layer_2 = nn.LSTM(2 * self.hidden_size, self.hidden_size, bidirectional=True)

        self.main_logits_layer = nn.Linear(10 * self.hidden_size, 1)

        self.main_lstm_layer_3 = nn.LSTM(14 * self.hidden_size, self.hidden_size, bidirectional=True)

        self.main_logits_layer_2 = nn.Linear(10 * self.hidden_size, 1)

    def forward(self, x, cx, x_mask, q, cq, q_mask, y, y2, device):
        """
        x -> [batch_size, max_sent_len]
        cx -> [batch_size, max_sent_len, word_len]
        q -> [batch_size, max_ques_len]
        cq -> [batch_size, max_ques_len, word_len]
        y -> [batch_size, max_sent_len]
        y2 -> [batch_size, max_sent_len]
        x_mask -> [batch_size, max_sent_len]
        q_mask -> [batch_size, max_ques_len]
        new_emb_mat -> [addtional_token_vocab_size, word_emb_len]
        """
        max_sent_len = x.size()[1]
        max_ques_len = q.size()[1]
        batch_size = x.size()[0]
        A_cx = self.char_emb(cx) # [batch_size, max_sent_len, max_word_len, char_emb_len]
        A_cq = self.char_emb(cq) # [batch_size, max_ques_len ,max_word_len, char_emb_len]
        A_cx = torch.reshape(A_cx, [batch_size, self.char_emb_len, max_sent_len, self.max_word_len])
        A_cq = torch.reshape(A_cq, [batch_size, self.char_emb_len, max_ques_len, self.max_word_len])

        if self.use_char_emb:
            xx = []
            qq = []
            for i in range(len(self.out_channel_dims)):
                xxc = self.conv1d_layers[i](torch.dropout(A_cx, p=self.keep_prob, train=self.training)) # [batch_size, out_channel_dims[i], max_sent_len, max_word_len - filter_height + 1]
                # torch.max along dimension will return max_elements and max_indices
                xx.append(torch.max(torch.relu(xxc), dim=3)[0]) # [batch_size, out_channel_dims[i], max_sent_len]
                if self.share_cnn_weights:
                    xxcq = self.conv1d_layers[i](torch.dropout(A_cq, p=self.keep_prob, train=self.training))
                    # torch.max along dimension will return max_elements and max_indices
                    qq.append(torch.max(torch.relu(xxcq), dim=3)[0])
                else:
                    xxcq = self.conv1d_layers_2[i](torch.dropout(A_cq, p=self.keep_prob, train=self.training))
                    # torch.max along dimension will return max_elements and max_indices
                    qq.append(torch.max(torch.relu(xxcq), dim=3)[0])
            xx = torch.reshape(torch.cat(xx, 1), [-1, max_sent_len, self.char_out_size]) # [batch_size, max_sent_len, char_out_size]
            qq = torch.reshape(torch.cat(qq, 1), [-1, max_ques_len, self.char_out_size]) # [batch_size, max_sent_len, char_out_size]

        if self.use_word_emb:
            A_x = self.word_emb(x) # [batch_size, max_sent_len, word_emb_len]
            A_q = self.word_emb(q) # [batch_size, max_ques_len, word_emb_len]
        
        if self.use_char_emb:
            xx = torch.cat([xx, A_x], dim=2) # [batch_size, max_sent_len, word_emb_len + char_out_size]
            qq = torch.cat([qq, A_q], dim=2) # [batch_size, max_ques_len, word_emb_len + char_out_size]
        else:
            xx = A_x
            qq = A_q
        
        if self.use_highway_network:
            # need to add weight decay here
            for i in range(self.highway_num_layers):
                xx_trans = torch.relu(self.highway_layers[i*2](torch.dropout(xx, p=self.keep_prob, train=self.training)))
                xx_gate = torch.sigmoid(self.highway_layers[i*2+1](torch.dropout(xx, p=self.keep_prob, train=self.training)))
                xx = xx_gate * xx_trans + (1 - xx_gate) * xx

                qq_trans = torch.relu(self.highway_layers[i*2](torch.dropout(qq, p=self.keep_prob, train=self.training)))
                qq_gate = torch.sigmoid(self.highway_layers[i*2+1](torch.dropout(qq, p=self.keep_prob, train=self.training)))
                qq = qq_gate * qq_trans + (1 - qq_gate) * qq
        
        xx = torch.reshape(xx, [max_sent_len, batch_size, -1]) # xx -> [batch_size, max_ques_len, word_emb_len + char_out_size]
        qq = torch.reshape(qq, [max_ques_len, batch_size, -1]) # qq -> [batch_size, max_ques_len, word_emb_len + char_out_size]
        
        # prepro
        x_lens = torch.sum(x_mask.int(), dim=1)
        q_lens = torch.sum(q_mask.int(), dim=1)

        u_h_0 = torch.zeros((2, batch_size, self.hidden_size)).to(device=device)
        u_c_0 = torch.zeros((2, batch_size, self.hidden_size)).to(device=device)

        output, _ = self.prepro_lstm_layer(qq, (u_h_0, u_c_0))
        output = torch.reshape(output, [batch_size, max_ques_len, -1])
        output = torch.dropout(output, p=self.keep_prob, train=self.training)
        u = output

        h_h_0 = torch.zeros((2, batch_size, self.hidden_size)).to(device=device)
        h_c_0 = torch.zeros((2, batch_size, self.hidden_size)).to(device=device)
        if self.share_lstm_weights:
            output, _ = self.prepro_lstm_layer(xx, (h_h_0, h_c_0))
        else:
            output, _ = self.prepro_lstm_layer_2(xx, (h_h_0, h_c_0))
        output = torch.reshape(output, [batch_size, max_sent_len, -1])
        h = output

        # main
        # attention layer (h_mask=x_mask, u_mask=q_mask)
        h_aug = torch.unsqueeze(h, 2).repeat(1, 1, max_ques_len, 1) # [batch_size, max_sent_len, max_question_len, 2 * hidden_size]
        u_aug = torch.unsqueeze(u, 1).repeat(1, max_sent_len, 1, 1) # [batch_size, max_sent_len, max_question_len, 2 * hidden_size]

        h_mask_aug = torch.unsqueeze(x_mask, 2).repeat(1, 1, max_ques_len) # [batch_size, max_sent_len, max_ques_len]
        u_mask_aug = torch.unsqueeze(q_mask, 1).repeat(1, max_sent_len, 1) # [batch_size, max_sent_len, max_ques_len]

        hu_mask = h_mask_aug & u_mask_aug

        hu_aug = h_aug * u_aug
        biattention_input = torch.cat([h_aug, u_aug, hu_aug], dim=3)
        biattention_output = self.biattention_layer(biattention_input)
        biattention_output = torch.squeeze(biattention_output) # [batch_size, max_sent_len, max_question_len]

        # exp_mask
        if hu_mask is not None:
            biattention_output = biattention_output + (1 - hu_mask.float()) * VERY_NEGATIVE_NUMBER

        u_a = torch.sum(u_aug * torch.softmax(torch.unsqueeze(biattention_output, dim=3), dim=2), dim=2) # [batch_size, max_sent_len, 2 * hidden_size]

        h_a = torch.sum(h * torch.softmax(torch.max(biattention_output, dim=2, keepdim=True)[0], dim=1), dim=1) # [batch_size, 2 * hidden_size]
        h_a = torch.unsqueeze(h_a, dim=1).repeat(1, max_sent_len, 1) # [batch_size, max_sent_len, 2 * hidden_size]

        # if tensor_dict is not None
        a_u = torch.softmax(biattention_output, dim=2)
        a_h = torch.softmax(torch.sum(biattention_output, dim=2), dim=1)

        p0 = torch.cat([h, u_a, h * u_a, h * h_a], dim=2) # [batch_size, max_sent_len, 8 * hidden_size]

        p0 = torch.reshape(p0, [max_sent_len, batch_size, -1])
        g0, _ = self.main_lstm_layer(p0) # [max_sent_len, batch_size, 2 * hidden_size]
        g0 = torch.dropout(g0, p=self.keep_prob, train=self.training)

        g1, _ = self.main_lstm_layer_2(g0) # [max_sent_len, batch_size, 2 * hidden_size]
        g1 = torch.dropout(g1, p=self.keep_prob, train=self.training)


        main_logits_input = torch.reshape(torch.cat([g1, p0], dim=2), [batch_size, max_sent_len, -1])
        main_logits_output = self.main_logits_layer(torch.dropout(main_logits_input, p=self.keep_prob, train=self.training))
        main_logits_output = torch.squeeze(main_logits_output) # [batch_size, max_sent_len]

        if x_mask is not None:
            main_logits_output = main_logits_output + (1 - x_mask.float()) * VERY_NEGATIVE_NUMBER

        a1i = torch.sum(torch.reshape(g1, [batch_size, max_sent_len, -1]) * torch.softmax(torch.unsqueeze(main_logits_output, dim=2), dim=1), dim=1) # [batch_size, 2 * hidden_size]

        a1i = torch.unsqueeze(a1i, dim=1).repeat(1, max_sent_len, 1) # [batch_size, max_sent_len, 2 * hidden_size]
        a1i = torch.reshape(a1i, [max_sent_len, batch_size, -1])

        main_lstm_3_input = torch.cat([p0, g1, a1i, g1 * a1i], dim=2)
        g2, _ = self.main_lstm_layer_3(main_lstm_3_input)

        main_logits_input_2 = torch.reshape(torch.cat([g2, p0], dim=2), [batch_size, max_sent_len, -1])
        main_logits_output_2 = self.main_logits_layer_2(torch.dropout(main_logits_input_2, p=self.keep_prob, train=self.training))
        main_logits_output_2 = torch.squeeze(main_logits_output_2) # [batch_size, max_sent_len]

        # yp = torch.softmax(main_logits_output, dim=1)
        # yp2 = torch.softmax(main_logits_output_2, dim=1)

        return main_logits_output, main_logits_output_2
        




