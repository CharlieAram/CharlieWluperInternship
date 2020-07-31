import torch
import torch.nn as nn
import torch.optim as optim
import pdb

from model1 import *


optimiser = optim.SGD(model.parameters(), lr=learning_rate)

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

#this returns an accuracy per batch
def accuracy(preds, y):

    #round up the initial predictions
    rounded_preds = torch.round(torch.sigmoid(preds))
    #calculate which are correct and output as float
    correct = (rounded_preds == y).float()
    #calculate accuracy
    acc = correct.sum() / len(correct)

    return acc


def trainer(model, iterator, optimiser, criterion):
    
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for i, batch in enumerate(iterator):
        optimiser.zero_grad()

        predictions = model(batch.text).squeeze(1)

        pdb.set_trace()

        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)
        #average precision score
        


        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        #give a visual indicator of how each epoch performs
        if i % 200 == 199:
            print(f'[{i}/{len(iterator)}] : epoch_acc: {epoch_acc / len(iterator):.2f} : epoch_loss: {epoch_loss / len(iterator)}')


    return epoch_loss / len(iterator), epoch_acc / len(iterator)