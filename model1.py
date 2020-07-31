import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess_data1 import *
from config1 import *



#the neural net model
class SentimentModel(nn.Module):

    #initialiser
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)

    #forward function
    def forward(self, text):

        embedded = self.embedding(text)

        output, hidden = self.rnn(embedded)


        hidden = F.relu(self.fc1(hidden[0]))

        return self.fc2(hidden.squeeze(0))


INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = 1

model = SentimentModel(INPUT_DIM, embedding_dim, hidden_dim, OUTPUT_DIM)


