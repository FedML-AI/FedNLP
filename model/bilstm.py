import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, embedding_length,
                 attention=False, embedding_weights=None):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention = attention
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(self.input_size, self.embedding_length)
        if embedding_weights is not None:
            self.word_embeddings.weights = nn.Parameter(embedding_weights, requires_grad=False)
        if self.attention:
            self.attention_layer = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)

        self.lstm_layer = nn.LSTM(self.embedding_length, self.hidden_size, self.num_layers, dropout=dropout, bidirectional=True)
        self.output_layer = nn.Linear(self.hidden_size * 2, self.output_size)

    def attention_forward(self, lstm_output, state):
        # We implement Luong attention here
        # lstm_output -> [batch_size, seq_len, num_directions*hidden_size]
        # state -> [batch_size, num_directions*hidden_size]

        hidden = state.unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights -> [batch_size, seq_len]
        soft_attn_weights = torch.softmax(attn_weights, 1)
        new_hidden = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # new_hidden ->[batch_size, num_directions*hidden_size]
        concat_hidden = torch.cat((new_hidden, state.squeeze(1)), 1)
        # concat_hidden ->[batch_size, 2*num_directions*hidden_size]
        output_hidden = self.attention_layer(concat_hidden)
        # output_hidden ->[batch_size, num_directions*hidden_size]
        return output_hidden

    def forward(self, input_seq, batch_size, device):
        # input_seq -> [batch_size, seq_len]
        input = self.word_embeddings(input_seq)
        # input -> [batch_size, seq_len, embedding_len]

        h_0 = torch.zeros((self.num_layers*2, batch_size, self.hidden_size)).to(device=device)
        c_0 = torch.zeros((self.num_layers*2, batch_size, self.hidden_size)).to(device=device)

        input = input.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(input, (h_0, c_0))
        # output -> [seq_len, batch_size, num_directions*hidden_size]

        output = output.permute(1, 0, 2)
        state = final_hidden_state.reshape((batch_size, self.num_layers, self.hidden_size * 2))[:, -1, :].squeeze(1)

        if self.attention:
            output = self.attention_forward(output, state)
        else:
            output = state

        logits = self.output_layer(output)

        return logits