import torch
from sklearn.metrics import classification_report, confusion_matrix

import pdb

from train1 import accuracy



def tester(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(iterator):

            predictions = model(batch.text).squeeze(1)

            pdb.set_trace()

            loss = criterion(predictions, batch.label)

            acc = accuracy(predictions, batch.label)

            conMatrix = confusion_matrix(batch.label.cpu().numpy(), predictions.cpu().numpy().astype(int))
            classifReport = classification_report(batch.label.cpu().numpy(), predictions.cpu().numpy().astype(int))

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), conMatrix, classifReport
