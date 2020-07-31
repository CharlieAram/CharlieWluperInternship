import torch
import time


from train1 import *
from test1 import *

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def trainModel():
    best_valid_loss = float('inf')

    for epoch in range(N_epoches):

        start_time = time.time()

        train_loss, train_acc = trainer(model, train_iterator, optimiser, criterion)
        valid_loss, valid_acc, valid_CM, valid_CF = tester(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'Sentiment-model1.pt')

        print(f'Epoch:  {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain  Loss: {train_loss: .3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tValid  Loss: {valid_loss: .3f} | Valid Acc: {valid_acc*100:.2f}%')


def testModel():
    model.load_state_dict(torch.load('Sentiment-model1.pt'))
    
    test_loss, test_acc, conMatrix, classifReport = tester(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    print(conMatrix)
    print(classifReport)

menu_choice = 0
while menu_choice != 2:
    print("\n\n -- Please choose a menu option -- \n")
    print("    0: Train NN")
    print("    1: Test NN")
    print("    2: Quit \n")
    
    try:
        menu_choice = int(input())
        print()
    except:
        print("Please try again")
        continue

    if menu_choice == 0:
        trainModel()
    elif menu_choice == 1:
        testModel()
    elif menu_choice == 2:
        print("\nprogram terminated\n\n")
    else:
        print("Please try again")