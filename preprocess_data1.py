
import torch
import string
from torchtext import data



def load_data(filepath, device, MAX_VOCAB_SIZE = 25_000):
    
    #standard tokenizer that removes punctuation and splits each sentance
    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()

    #intializing of data fields: TEXT contains each movie review, LABEL contains whether it is positive or not
    TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=100)
    LABEL = data.Field(sequential=False, use_vocab=False)

    
    print("data loading...")

    #this is the formats/fields data will be stored in
    datafields = [("text", TEXT), ("label", LABEL)]



    #we split the dataset into train, dev and test
    train, valid, test = data.TabularDataset.splits(path=filepath, 
                                                    train="Train1.csv", 
                                                    validation="Valid1.csv", 
                                                    test="Test1.csv", 
                                                    format="csv", 
                                                    skip_header=True, 
                                                    fields=datafields)

    
    print("building vocab...")
    #numericalise vocab of TEXT through torchtext, this will allow following operations to be done
    TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)

    print('constructing iterators...')

    train_iterator = data.BucketIterator(train, device=device, batch_size=32, sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)
    
    valid_iterator = data.BucketIterator(valid, device=device, batch_size=32, sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)
    
    test_iterator = data.BucketIterator(test, device=device, batch_size=32, sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)

    print("data load successful")

    return TEXT, LABEL, train, valid, test, train_iterator, valid_iterator, test_iterator




#this allows for GPU acceleration if cuda is installed, but if not it will be a CPU operation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#calls load_data function and assigns variables, the first argument is empty because the data is in the same file path as the script

TEXT, LABEL, train, valid, test, train_iterator, valid_iterator, test_iterator = load_data('', device)


