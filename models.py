import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module): # Average word vector of entire data entry
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors) #Convert token into word vector 
        self.fc = nn.Linear(embedding_dim, 5)
        
    def forward(self, x, lengths=None):
        embedded = self.embedding(x) # [sentence length, batch size, embedding_dim]
        average = embedded.mean(0) 
        output = self.fc(average).squeeze(1)
        return output

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim): 
        super(RNN, self).__init__()

        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim 
        
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors) # Encode word vectors         
        self.gru = nn.GRU(embedding_dim, hidden_dim) # instantiate RNN (GRU) 
        self.fc = nn.Linear(hidden_dim, 5) # Uses last hidden state to generate output 

    def forward(self, x, lengths=None):
        embedded = self.embedding(x) # embedded = [sentence_length, batch_size, embedding_dim]
        pack_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=False)
        _, hidden = self.gru(pack_embedded)
        output = self.fc(hidden)[-1].squeeze()
        return output
