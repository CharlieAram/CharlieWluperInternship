import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

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

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers*2, minibatch_size, hidden_dim)
        # the *2 is because this is a bi-LSTM
        self.hidden = (torch.zeros(4, batch_size, self.hidden_dim),
                torch.zeros(4, batch_size, self.hidden_dim))

    
    #forward function
    def forward(self, text):

        embedded = self.embedding(text)

        embedded = embedded.permute(1,0,2)

        output, hidden = self.rnn(embedded, self.hidden)

        #print(hidden[0].shape)

        hidden = F.relu(self.fc1(hidden[0]))

        #print(hidden.shape)

        #print(embedded.shape)

        #print(text.shape)

        #pdb.set_trace()

        return self.fc2(hidden.sum(dim=0).squeeze(1))


INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = 1

model = SentimentModel(INPUT_DIM, embedding_dim, hidden_dim, OUTPUT_DIM)


