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

#Parameters
lr=0.001
#full_size=2700
batch_size= 1024
patience = 15
n_epochs = 100
seed = 10
dropout=0.2
folder_name = str('CNN_1-16-32-6_lr-' + str(lr) + '_batch-' + str(batch_size) +'_n_epochs-' + str(n_epochs)+'_dropout-'+str(dropout))
save_file_name = str(folder_name +'.pth')
torch.manual_seed(seed)
letters=['Gamma', 'Beta', 'Eta', 'Phi', 'Theta', 'Zeta']
le = preprocessing.LabelEncoder()
le.fit(letters)
config_model = {}
directory = os.path.expanduser('~/Documents/MSc/DeepLearning/Saved_models/' + folder_name)

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            Dropout(dropout),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            Dropout(dropout),
            MaxPool2d(kernel_size=2, stride=2),
            
            
        )

        self.linear_layers = Sequential(
            Linear(7 * 7 * 32, 6),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def gather_csv():
    def find_matching_string(input_string, string_list):
        for string in string_list:
            if string in input_string:
                return string
        return None  

    dir_path1 = "Nikolas_Train"
    dir_path2 = "Sofia_Train"
    dir_val_path1 = "Nikolas_Val"
    dir_val_path2 = "Sofia_Val"
    
    # csv files in the path
    files = glob.glob(dir_path1 + "/*.csv")
    files2 = glob.glob(dir_path2 + "/*.csv")
    
    files_val1 = glob.glob(dir_val_path1 + "/*.csv")
    files_val2 = glob.glob(dir_val_path2 + "/*.csv")
    
    # defining an empty list to store 
    # content
    content = []
    content2 = []
    label_list = []
    
    content_val1 = []
    content_val2 = []
    label_val = []
    
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
    
    
    #Validation folder
    for filename in files_val1:
        matching_string = find_matching_string(filename, letters)
            
        # reading content of csv file
        # content.append(filename)
        with open(filename, newline='') as csvfile:
            data = list(csv.reader(csvfile))
            content_val1.append(data)
            print(len(content_val1))
            if matching_string:
                for i in range(len(data)):
                    label_val.append(matching_string)
            else:
                print(filename)
        
    reshaped_data_val1 = (np.array(content_val1)).reshape(50*len(files_val1), 784)
    
    for filename in files_val2:
        matching_string = find_matching_string(filename, letters)
            
        # reading content of csv file
        # content.append(filename)
        with open(filename, newline='') as csvfile:
            data = list(csv.reader(csvfile))
            content_val2.append(data)
            if matching_string:
                for i in range(len(data)):
                    label_val.append(matching_string)
            else:
                print(filename)
                
    reshaped_data_val2 = (np.array(content_val2)).reshape(100*len(files_val2), 784)
    
    reshaped_data_val = np.vstack((reshaped_data_val1, reshaped_data_val2))
    label_val=le.fit_transform(label_val)
    
    return reshaped_data.astype(np.float32), np.array(label_list), reshaped_data_val.astype(np.float32), label_val

    


def save_files(plt_json):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = os.path.join(directory, 'config_model.json')
    plot_path = os.path.join(directory, 'learning_curve_')
    print('Saving...')
    try:
        with open(file_path, 'w') as outfile:
            json.dump(config_model, outfile, indent=4)
            for i, plt in enumerate(plt_json['plot']):
                plt.savefig(plot_path+plt_json['name'][i]+'.png')
        print('Saved files successfully!')
    except Exception as e:
        print('Error while saving!')
        print(str(e))
        return

def train_model(model,criterion,optimizer,train_loader,valid_loader,save_file_name,
          max_epochs_stop=3,n_epochs=20,print_every=1):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
        
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    valid_f1_max = 0
    
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        train_f1_score = 0
        train_f1_score_macro = 0
        valid_f1_score = 0
        valid_f1_score_macro = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):


            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Calculate f1 macro averaged score
            print('\n Pred:', pred)
            print('\n Target:', target)
            f1_score = multiclass_f1_score(target, pred, num_classes=6, average=None)
            train_f1_score += f1_score * data.size(0)

            f1_score_macro = multiclass_f1_score(target, pred, num_classes=6, average="macro")
            train_f1_score_macro += f1_score_macro * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            print("Start validation")
            print(
              f'{timer() - start:.2f} seconds.')
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                    # Calculate f1 macro averaged score
                    f1_score = multiclass_f1_score(target, pred, num_classes=6, average=None)
                    valid_f1_score += f1_score * data.size(0)
                    
                    f1_score_macro = multiclass_f1_score(target, pred, num_classes=6, average="macro")
                    valid_f1_score_macro += f1_score_macro * data.size(0)
                    
                    for i in range(6):
                        if i not in list(pred):
                            print(f'\n {le.inverse_transform([i])} does not exist in pred')
                        if i not in list(target):
                            print(f'\n {le.inverse_transform([i])} does not exist in target')
                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                # Calculate average f1 score
                train_f1_score = train_f1_score / len(train_loader.dataset)
                valid_f1_score = valid_f1_score / len(valid_loader.dataset)

                train_f1_score_macro = train_f1_score_macro / len(train_loader.dataset)
                valid_f1_score_macro = valid_f1_score_macro / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc, train_f1_score, valid_f1_score, train_f1_score_macro, valid_f1_score_macro])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )


                # Save the model if validation loss decreases
                if valid_f1_score_macro > valid_f1_max:
                    # Save model
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torch.save(model.state_dict(), directory+'/'+save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    train_loss_min = train_loss
                    train_best_acc = train_acc
                    train_f1_score_best = train_f1_score
                    valid_f1_score_best = valid_f1_score
                    train_f1_score_macro_best = train_f1_score_macro
                    valid_f1_score_macro_best = valid_f1_score_macro

                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        model.load_state_dict(torch.load(directory+'/'+save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer
                        config_model.update({'Early Stopping': 'yes', 'total_time': total_time, 'best_epoch': best_epoch, \
                                             'last_epoch': epoch, 'valid_loss_min': valid_loss_min, 'valid_best_acc': valid_best_acc, \
                                             'train_loss_min': train_loss_min, 'train_best_acc': train_best_acc, \
                                             'train_f1_score_best': train_f1_score_best.tolist(), 'valid_f1_score_best': valid_f1_score_best.tolist(), \
                                             'train_f1_score_macro_best': train_f1_score_macro_best.tolist(), 'valid_f1_score_macro_best': valid_f1_score_macro_best.tolist()})
                        count = 0
                        for i in valid_f1_score_best:
                          config_model.update({letters[count]:i.item()})
                          count += 1

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc', 'train_f1_score', 'valid_f1_score', 'train_f1_score_macro', 'valid_f1_score_macro'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / epoch:.2f} seconds per epoch.'
    )

    config_model.update({'Early Stopping': 'no', 'total_time': total_time, 'best_epoch': best_epoch, \
                                             'last_epoch': epoch, 'valid_loss_min': valid_loss_min, 'valid_best_acc': valid_best_acc, \
                                             'train_loss_min': train_loss_min, 'train_best_acc': train_best_acc, \
                                             'train_f1_score_best': train_f1_score_best.tolist(), 'valid_f1_score_best': valid_f1_score_best.tolist(), \
                                             'train_f1_score_macro_best': train_f1_score_macro_best.tolist(), 'valid_f1_score_macro_best': valid_f1_score_macro_best.tolist()})
    count = 0
    for i in valid_f1_score_best:
        config_model.update({letters[count]:i.item()})
        count += 1
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'train_f1_score', 'valid_f1_score', 'train_f1_score_macro', 'valid_f1_score_macro'])

    return model, history


def main():
    train_data_array, train_label_list, val_data_array, val_label_list = gather_csv()

    
    train_x = np.array(train_data_array)
    train_y = np.array(train_label_list)
    val_x = np.array(val_data_array)
    val_y= np.array(val_label_list)
    print('Len train_x:', len(train_x))
    print('Len val_x:', len(val_x))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_x = torch.from_numpy(train_x.reshape(train_x.shape[0], 1, 28, 28)).to(device)
    train_y = torch.from_numpy(train_y.astype(int)).to(device)
    val_x = torch.from_numpy(val_x.reshape(val_x.shape[0], 1, 28, 28)).to(device)
    val_y = torch.from_numpy(val_y.astype(int)).to(device)
    
    train = TensorDataset(train_x.to(device), train_y.long().to(device))
    val = TensorDataset(val_x.to(device), val_y.long().to(device))

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(train, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val, batch_size=1800, shuffle=True),
    }

    model = Net()
    optimizer = Adam(model.parameters(), lr=lr)
    #weight_decay=1e-5
    criterion = CrossEntropyLoss()
    
    model, history = train_model(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    dataloaders['val'],
    save_file_name=save_file_name,
    max_epochs_stop=patience,
    n_epochs=n_epochs,
    print_every=10)
    
    plt_json = {}
    name = []
    plot = []
    

    # Plot 1: Training and Validation Losses
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        ax1.plot(history[c], label=c)
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Negative Log Likelihood')
    ax1.set_title('Training and Validation Losses')
    name.append('loss')
    plot.append(fig1)

    # Plot 2: Training and Validation Accuracy
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for c in ['train_acc', 'valid_acc']:
        ax2.plot(100 * history[c], label=c)
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    name.append('Accuracy')
    plot.append(fig2)

    # Plot 3: Training and Validation F1 score macro
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for c in ['train_f1_score_macro', 'valid_f1_score_macro']:
        ax3.plot(100 * history[c], label=c)
    ax3.legend()
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average F1 score macro')
    ax3.set_title('Training and Validation F1 score macro')
    name.append('F1 macro')
    plot.append(fig3)
    
    plt_json.update({'name': name, 'plot': plot})
    
    save_files(plt_json)

if __name__ == "__main__":
    main()
