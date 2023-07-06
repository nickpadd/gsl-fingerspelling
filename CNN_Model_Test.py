import torch
from torch.autograd import Variable
from torch.nn import (
    Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax,
    BatchNorm2d, Dropout
)
from torch.optim import Adam
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import csv
from sklearn import preprocessing
import glob
import os
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import multiclass_f1_score
from MyCNN_batch import Net


seed = 10
folder_name = str('Best_CNN_model_test')
torch.manual_seed(seed)
letters=['Gamma', 'Beta', 'Eta', 'Phi', 'Theta', 'Zeta']
le = preprocessing.LabelEncoder()
le.fit(letters)
config_model = {}
directory = os.path.expanduser('~/Documents/MSc/DeepLearning/Saved_models/' + folder_name)



def gather_csv():
    def find_matching_string(input_string, string_list):
        for string in string_list:
            if string in input_string:
                return string
        return None  

    dir_path1 = "Nikolas_Test"
    dir_path2 = "Sofia_Test"
    
    # csv files in the path
    files = glob.glob(dir_path1 + "/*.csv")
    files2 = glob.glob(dir_path2 + "/*.csv")
    
    # defining an empty list to store 
    # content
    content = []
    content2 = []
    label_list = []
    
    
    # checking all the csv files in the 
    # specified path
    for filename in files:
        matching_string = find_matching_string(filename, letters)
            
        # reading content of csv file
        # content.append(filename)
        with open(filename, newline='') as csvfile:
            data = list(csv.reader(csvfile))
            content.append(data)
            print(len(content))
            if matching_string:
                for i in range(len(data)):
                    label_list.append(matching_string)
            else:
                print(filename)
        
    reshaped_data1 = (np.array(content)).reshape(50*len(files), 784)
    
    for filename in files2:
        matching_string = find_matching_string(filename, letters)
            
        # reading content of csv file
        # content.append(filename)
        with open(filename, newline='') as csvfile:
            data = list(csv.reader(csvfile))
            content2.append(data)
            if matching_string:
                for i in range(len(data)):
                    label_list.append(matching_string)
            else:
                print(filename)
                
    reshaped_data2 = (np.array(content2)).reshape(100*len(files2), 784)
    
    reshaped_data = np.vstack((reshaped_data1, reshaped_data2))
    label_list=le.fit_transform(label_list)
    
    return reshaped_data.astype(np.float32), np.array(label_list)


def save_files(config_model):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = os.path.join(directory, 'config_model.json')
    print('Saving...')
    try:
        with open(file_path, 'w') as outfile:
            json.dump(config_model, outfile, indent=4)
        print('Saved files successfully!')
    except Exception as e:
        print('Error while saving!')
        print(str(e))
        return

def test_model(model, criterion, test_loader):
        
    valid_loss_min = np.Inf
    valid_f1_max = 0
    
    history = []

    # keep track of training and validation loss each epoch
    test_loss = 0.0
    test_acc = 0
    test_f1_score = 0
    test_f1_score_macro = 0

    print("Starting Test")

    # Don't need to keep track of gradients
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()

        # Validation loop
        for data, target in test_loader:

            # Forward pass
            output = model(data)

            # Validation loss
            loss = criterion(output, target)
            # Multiply average loss times the number of examples in batch
            test_loss += loss.item() * data.size(0)

            # Calculate validation accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples
            test_acc += accuracy.item() * data.size(0)

            # Calculate f1 macro averaged score
            f1_score = multiclass_f1_score(target, pred, num_classes=6, average=None)
            test_f1_score += f1_score * data.size(0)
                    
            f1_score_macro = multiclass_f1_score(target, pred, num_classes=6, average="macro")
            test_f1_score_macro += f1_score_macro * data.size(0)
                    
            for i in range(6):
                if i not in list(pred):
                    print(f'\n {le.inverse_transform([i])} does not exist in pred')
                if i not in list(target):
                    print(f'\n {le.inverse_transform([i])} does not exist in target')
        # Calculate average losses
        test_loss = test_loss / len(test_loader.dataset)

        # Calculate average accuracy
        test_acc = test_acc / len(test_loader.dataset)

        # Calculate average f1 score   
        test_f1_score = test_f1_score / len(test_loader.dataset)

        test_f1_score_macro = test_f1_score_macro / len(test_loader.dataset)

        history.append([test_loss, test_acc, test_f1_score, test_f1_score_macro])


        print(
            f'\t\tTest Accuracy: {100 * test_acc:.2f}%\t Test f1 macro: {100 * test_f1_score_macro:.2f}%'
        )

    config_model.update({
    'test_loss_min': test_loss,
    'test_best_acc': test_acc,
    'test_f1_score': test_f1_score.tolist(),
    'test_f1_score_macro': test_f1_score_macro.tolist()
    })

    count = 0
    for i in test_f1_score:
        config_model.update({letters[count]:i.item()})
        count += 1
        
    
    save_files(config_model)
    # Format history
    history = pd.DataFrame(
        history,
        columns=['test_loss', 'test_acc', 'test_f1_score', 'test_f1_score_macro'])

    return model, history


def main():
    test_data_array, test_label_list = gather_csv()

    
    test_x = np.array(test_data_array)
    test_y = np.array(test_label_list)

    print('Len test_x:', len(test_x))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    test_x = torch.from_numpy(test_x.reshape(test_x.shape[0], 1, 28, 28)).to(device)
    test_y = torch.from_numpy(test_y.astype(int)).to(device)
    
    test = TensorDataset(test_x.to(device), test_y.long().to(device))

    # Dataloader iterators
    dataloaders = {
        'test': DataLoader(test, batch_size=1000, shuffle=True)
    }

    model = Net()
    model.load_state_dict(torch.load('Saved_models/4_CNN_1-16-32-6_lr-0.001_batch-1024_n_epochs-100_dropout-0/CNN_1-16-32-6_lr-0.001_batch-1024_n_epochs-100_dropout-0.pth'))
    model.eval()
    criterion = CrossEntropyLoss()
    
    model, history = test_model(
    model,
    criterion,
    dataloaders['test'])


if __name__ == "__main__":
    main()
