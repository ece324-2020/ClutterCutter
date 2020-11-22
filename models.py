import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):  # Scalar output for each sentence (probability of being subjective)

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)  # Convert token into word vector
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # x = [sentence length, batch size (bs)] <-- x's shape
        embedded = self.embedding(x)  # [sentence length, batch size, embedding_dim]
        average = embedded.mean(0)
        output = self.fc(average).squeeze(1)  # [bs]

        return output

class RNN(nn.Module):

    def __init__(self, embedding_dim, vocab, hidden_dim):  # 100,100
        super(RNN, self).__init__()

        self.embedding_dim = embedding_dim  # 100
        self.hidden_dim = hidden_dim  # 100

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)  # Matrix of word vectors
        self.gru = nn.GRU(embedding_dim, hidden_dim)  # instantiate RNN (GRU)

        self.fc = nn.Linear(hidden_dim, 1)  # [100,1] Takes in last hidden state to generate output
        self.af = nn.Sigmoid()  # Activation function for probability output

    def init_hidden(self, batch_size):  # Generate initial hidden state
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        return h0

    def forward(self, x, x_len, lengths=None):
        # x = [sentence length, batch size (bs)] <-- x's shape
        batch_size = (x.T).shape[0]  # A single number

        hidden = self.init_hidden(batch_size)  # Create the initial hidden state
        embedded = self.embedding(x)  # embedded = [bs, sentence length, embedding_dim]

        pack_embedded = nn.utils.rnn.pack_padded_sequence(embedded, x_len, batch_first=False)

        pack_output, hidden = self.gru(pack_embedded, hidden)  ###

        hidden = hidden.contiguous().view(-1, self.hidden_dim)  # [1, bs, 1] --> [bs, 1]

        fc_outputs = self.fc(hidden)  # [bs,1] <--- use last hidden as linear function's input

        outputs = self.af(fc_outputs)  # [bs,1] <--- make it a probability

        return outputs.flatten()