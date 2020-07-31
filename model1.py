import torch
import torch.nn as nn
import pdb
from preprocess_data1 import *
from config1 import *

#the neural net model
class SentimentModel(nn.Module):

    #initialiser
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)


    #forward function
    def forward(self, text):

        embedded = self.embedding(text)

        output, hidden = self.rnn(embedded)

        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = 1

model = SentimentModel(INPUT_DIM, embedding_dim, hidden_dim, OUTPUT_DIM)


