import torch
from torch.autograd import Variable
from torch.nn import (
    Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax,
    BatchNorm2d, Dropout
)
from torch.optim import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import csv
from sklearn import preprocessing
import glob
import matplotlib.pyplot as plt


class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(7 * 7 * 32, 7)
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

    dir_path = "CSVfiles"
    # csv files in the path
    files = glob.glob(dir_path + "/*.csv")
    
    # defining an empty list to store 
    # content
    content = []
    letters=['Gamma', 'Beta', 'Eta', 'Phi', 'Theta', 'Xi', 'Zeta']
    le = preprocessing.LabelEncoder()
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
            if matching_string:
                for i in range(len(data)):
                    label_list.append(matching_string)
        
    reshaped_data = (np.array(content)).reshape(50*len(files), 784)
    label_list=le.fit_transform(label_list)
    return reshaped_data.astype(np.float32), np.array(label_list)

    
def evaluate_model(n_model, data, labels):
    # prediction for training set
    device = next(n_model.parameters()).device
    with torch.no_grad():
        output = n_model(data.to(device))
    softmax = torch.exp(output)
    #prob = list(softmax.numpy())
    predictions = torch.argmax(softmax, -1)
    return f1_score(labels.cpu().data.numpy(), predictions.cpu().data.numpy(), average='macro')



def train(model, X_train, y_train, X_val, y_val, optimizer, criterion, n_epochs):

    train_losses, val_losses, train_f1, val_f1 = [], [], [], []
    best_f1 = 0.0
    for e in range(n_epochs):
        model.train()
        tr_loss = 0
        x_train, y_train = Variable(X_train), Variable(y_train)
        x_val, y_val = Variable(X_val), Variable(y_val)


        optimizer.zero_grad()      # clear the Gradients
        output_train = model(x_train)   # prediction for training set
        output_val = model(x_val)   # prediction for validation set

        # compute losses
        loss_train = criterion(output_train, y_train)
        loss_val = criterion(output_val, y_val)
        train_losses.append(loss_train.cpu().detach().numpy())
        val_losses.append(loss_val.cpu().detach().numpy())

        # backprop and update weights:
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        print('Epoch : ', e+1, '\t', 'loss :', loss_val)
        train_f1.append(evaluate_model(model, X_train, y_train))
        val_f1.append(evaluate_model(model, X_val, y_val))
        if  evaluate_model(model, X_val, y_val) > best_f1: 
            best_f1 = evaluate_model(model, X_val, y_val)
            print('New best f1: \t', best_f1)
            torch.save(model.state_dict(), 'best-model_MyCNN-parameters.pt')
        
    plt.subplot(2,1,1)
    plt.plot(range(n_epochs), train_losses, '--', label='Train Loss')
    plt.plot(range(n_epochs), val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(range(n_epochs), train_f1, '--', label='Train F1')
    plt.plot(range(n_epochs), val_f1, label='Validation F1')
    plt.title('F1')
    plt.legend()

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()


def main():
    data_array, label_list=gather_csv()
    
    print(data_array)
    print(label_list)
    train_x, val_x, train_y, val_y= train_test_split(data_array, label_list, test_size = 0.1, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # rea

    train_x = torch.from_numpy(train_x.reshape(train_x.shape[0], 1, 28, 28)).to(device)
    train_y = torch.from_numpy(train_y.astype(int)).to(device)
    val_x = torch.from_numpy(val_x.reshape(val_x.shape[0], 1, 28, 28)).to(device)
    val_y = torch.from_numpy(val_y.astype(int)).to(device)

    model = Net()
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()
    train(model, train_x, train_y, val_x, val_y, optimizer, criterion, n_epochs=50)



if __name__ == "__main__":
    main()
