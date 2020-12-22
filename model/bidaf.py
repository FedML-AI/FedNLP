import torch
from torch import nn

class BIDAF_SpanExtraction(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, char_emb_length, word_emb_length, filter_sizes, heights, 
    share_cnn_weights, max_word_length, char_out_size, keep_prob, share_lstm_weights, hidden_size,
    use_char_emb, 
    use_word_emb, 
    use_glove_for_unk,
    use_highway_network,
    char_emb_weights=None, word_emb_weights=None, new_emb_weights=None, highway_num_layers=0):
        super(BIDAF_SpanExtraction, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.char_emb_length = char_emb_length
        self.word_emb_length = word_emb_length
        self.filter_sizes = filter_sizes
        self.heights = heights
        self.share_cnn_weights = share_cnn_weights
        self.max_word_length = max_word_length
        self.char_out_size = char_out_size
        self.keep_prob = keep_prob
        self.highway_num_layers = highway_num_layers
        self.share_lstm_weights = share_lstm_weights
        self.hidden_size = hidden_size
        intput_size = 0
        if use_char_emb:
            if char_emb_weights:
                self.char_emb = nn.Embedding.from_pretrained(torch.tensor(char_emb_weights))
            else:
                self.char_emb = nn.Embedding(self.char_vocab_size, self.char_emb_length)
            input_size += self.char_emb_length
        if use_word_emb:
            if word_emb_weights:
                self.word_emb = nn.Embedding.from_pretrained(torch.tensor(word_emb_weights))
            else:
                self.word_emb = nn.Embedding(self.word_vocab_size, self.word_emb_length)
            input_size += self.word_emb_length
        if use_glove_for_unk:
            self.new_emb = nn.Embedding.from_pretrained(torch.tensor(new_emb_weights))
            input_size += self.new_emb_weights.size()[-1]
        self.conv1d_layers = nn.ModuleList()
        for filter_size, height in zip(filter_sizes, heights):
            self.conv1d_layers.append(nn.Conv1d(self.char_emb_length, height, filter_size))
        if not share_cnn_weights:
            self.conv1d_layers_2 = nn.ModuleList()
            for filter_size, height in zip(filter_sizes, heights):
                self.conv1d_layers_2.append(nn.Conv1d(self.char_emb_length, height, filter_size))
        
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

    def forward(self, x, cx, x_lens, q, cq, q_lens, y, y2, device):
        """
        x -> [num_docs, num_paragraphs, seq_length]
        cx -> [num_docs, num_paragraphs, seq_length, word_length]
        p -> [num_docs, num_paragraphs]
        q -> [num_questions, seq_length]
        cq -> [num_questions, seq_length, word_length]
        y -> [num_questions, [(sent_idx, start_word_idx),(sent_idx, end_word_idx)]]
        rx(*x)(*p) -> [num_questions, [doc_id, paragraph_id]]
        rcx(*cs) -> [num_questions, [doc_id, paragraph_id]]
        cy -> [num_questions, [(0, length of last word in answer text)]
        idxs -> [num_questions] index from 0 to num_questions
        ids -> [num_questions] each question's id
        answerss -> [num_questions] answers in text format
        """
        max_sent_length = x.size()[2]
        max_ques_length = q.size()[1]
        max_num_sents = x.size()[1]
        batch_size = x.size()[0]
        A_cx = self.char_emb(cx) # [batch_size, max_num_sents, max_seq_length, max_word_length, char_emb_length]
        A_cq = self.char_emb(cq) # [batch_size, max_num_questions ,max_word_length, char_emb_length]
        A_cx = torch.reshape(A_cx, [-1, max_sent_length, self.max_word_size, self.char_emb_length])
        A_cq = torch.reshape(A_cq, [-1, max_sent_length, self.max_word_length, self.char_emb_length])

        if self.use_char_emb:
            xx = []
            qq = []
            for i in enumerate(len(self.filter_sizes)):
                xxc = self.conv1d_layers[i](torch.dropout(A_cx, p=self.keep_prob))
                xx.append(torch.max(torch.relu(xxc), dim=2))
                if self.share_cnn_weights:
                    xxcq = self.conv1d_layers[i](torch.dropout(A_cq, p=self.keep_prob))
                    qq.append(torch.max(torch.relu(xxcq), dim=2))
                else:
                    xxcq = self.conv1d_layers[i](torch.dropout(A_cq, p=self.keep_prob))
                    qq.append(torch.max(torch.relu(xxcq), dim=2))
            xx = torch.reshape(torch.cat(xx, 2), [-1, max_num_sents, max_sent_length, self.char_out_size])
            qq = torch.reshape(torch.cat(qq, 2), [-1, max_ques_length, self.char_out_size])

        if self.use_word_emb:
            A_x = self.word_emb(x)
            A_q = self.word_emb(q)
            if self.use_glove_for_unk:
                A_x = torch.cat([A_x, self.new_emb(x)], dim=0)
                A_q = torch.cat([A_q, self.new_emb(q)], dim=0)
        
        if self.use_char_emb:
            xx = torch.cat([xx, A_x], dim=3)
            qq = torch.cat([qq, A_q], dim=2)
        else:
            xx = A_x
            qq = A_q
        
        if self.use_highway_network:
            # need to add weight decay here
            for i in range(self.highway_num_layers):
                xx_trans = torch.relu(self.highway_layers[i*2](torch.dropout(xx, p=self.keep_prob)))
                xx_gate = torch.sigmoid(self.highway_layers[i*2+1](torch.dropout(xx, p=self.keep_prob)))
                xx = xx_gate * xx_trans + (1 - xx_gate) * xx

                qq_trans = torch.relu(self.highway_layers[i*2](torch.dropout(qq, p=self.keep_prob)))
                qq_gate = torch.sigmoid(self.highway_layers[i*2+1](torch.dropout(qq, p=self.keep_prob)))
                qq = qq_gate * qq_trans + (1 - qq_gate) * qq
        
        # prepro
        u_h_0 = torch.zeros((2, batch_size, self.hidden_size)).to(device=device)
        u_c_0 = torch.zeros((2, batch_size, self.hidden_size)).to(device=device)

        output, _ = self.prepro_lstm_layer(qq, (u_h_0, u_c_0))

        u = torch.cat([output[i, q_len-1, :].unsqueeze(0) for i, q_len in enumerate(q_lens)], dim=0)

        h_h_0 = torch.zeros((2, batch_size, self.hidden_size)).to(device=device)
        h_c_0 = torch.zeros((2, batch_size, self.hidden_size)).to(device=device)
        if self.share_lstm_weights:
            output, _ = self.prepro_lstm_layer(xx, (h_h_0, h_c_0))
        else:
            output, _ = self.prepro_lstm_layer_2(xx, (h_h_0, h_c_0))

        h = torch.cat([output[i, x_len-1, :].unsqueeze(0) for i, x_len in enumerate(x_lens)], dim=0)

        return u, h