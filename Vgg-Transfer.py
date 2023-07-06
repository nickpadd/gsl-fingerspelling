from torchvision import transforms, datasets, models
import torch
from torch.nn import (
    Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax,
    BatchNorm2d, Dropout
)
from torch import optim, cuda
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
import glob
from PIL import Image
from sklearn import preprocessing
from torchvision.models import VGG16_Weights
from torch.optim import Adam
from timeit import default_timer as timer
from torchsummary import summary

# Data science tools
import numpy as np
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt


config_model = {}
    
def train_model(model,criterion,optimizer,train_loader,valid_loader,save_file_name,
          max_epochs_stop=3,n_epochs=20,print_every=1):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): vgg to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pth'): file path to save the model
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained vgg with best weights
        history (DataFrame): history of train and validation loss, accuracy and f1_score
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_f1_macro_max = 0
    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    for epoch in range(n_epochs):

        # keep track of training and validation loss for each epoch
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

            # Calculate f1 macro average score
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

                    # Calculate f1 macro average score
                    f1_score = multiclass_f1_score(target, pred, num_classes=6, average=None)
                    valid_f1_score += f1_score * data.size(0)

                    f1_score_macro = multiclass_f1_score(target, pred, num_classes=6, average="macro")
                    valid_f1_score_macro += f1_score_macro * data.size(0)

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


                # Save the model if validation f1_score increases
                if valid_f1_score_macro > valid_f1_macro_max:
                    # Save model
                    print('save model')
                    torch.save(model, save_file_name)

                    # Track improvement
                    epochs_no_improve = 0
                    valid_f1_macro_max = valid_f1_score_macro
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    train_loss_min = train_loss
                    train_best_acc = train_acc
                    train_f1_score_best = train_f1_score
                    valid_f1_score_best = valid_f1_score
                    train_f1_score_macro_best = train_f1_score_macro

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

                        # Load the best model
                        model = torch.load(save_file_name)

                        # Attach the optimizer
                        model.optimizer = optimizer
                        config_model.update({'Early Stopping': 'yes', 'total_time': total_time, 'best_epoch': best_epoch, \
                                             'last_epoch': epoch, 'valid_loss_min': valid_loss_min, 'valid_best_acc': valid_best_acc, \
                                             'train_loss_min': train_loss_min, 'train_best_acc': train_best_acc, \
                                             'train_f1_score_macro_best': train_f1_score_macro_best, 'valid_f1_macro_max': valid_f1_macro_max})
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
                          'last_epoch': n_epochs, 'valid_loss_min': valid_loss_min, 'valid_best_acc': valid_best_acc, \
                          'train_loss_min': train_loss_min, 'train_best_acc': train_best_acc, \
                          'train_f1_score_macro_best': train_f1_score_macro_best, 'valid_f1_macro_max': valid_f1_macro_max})
    count = 0
    for i in valid_f1_score_best:
        config_model.update({letters[count]:i.item()})
        count += 1
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'train_f1_score', 'valid_f1_score', 'train_f1_score_macro', 'valid_f1_score_macro'])

    return model, history


def gather_png(dir_path):
    """ input: path of train/validation/test set
        process:
            - load all .png files included in path
            - match letters with filename string and keep only those
              filenames which contain the letters we are interesting for
            - load images (selected png files)
            - preprocess image and append result in content list
        output: list of preprocessed images and np array with labels for each image
    """
    def find_matching_string(input_string, string_list):
        for string in string_list:
            if string in input_string:
                return string
        return None

    preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    # csv files in the path
    files = glob.glob(dir_path + "/*.png")

    # defining an empty list to store
    content = []
    # define the letters we examine (classes)
    letters = ['Gamma', 'Beta', 'Eta', 'Phi', 'Theta', 'Zeta']
    le = preprocessing.LabelEncoder()
    le.fit(letters)
    label_list = []

    for filename in files:
        matching_string = find_matching_string(filename, letters)

        with open(filename, 'rb') as image:
            data = Image.open(image)
            input_tensor = preprocess(data)

            content.append(input_tensor)
            if matching_string:
                label_list.append(matching_string)

    label_list = le.fit_transform(label_list)
    content = torch.stack(content)
    print('Dataset size:', content.size)
    print('Lables size:', len(label_list))

    return content, np.array(label_list, dtype=np.int32)


def get_pretrained_model(model_name, n_classes):
    """ input: model and number of classes
        process:
            - only if model is vgg16, load model with weights VGG16_Weights.IMAGENET1K_V1
            - freeze early layers
            - load features of 6th classifier
            - add a custom 6th classifier, using nn.Sequential() module and
              specifying extra layers one after the other
        output: return the model
    """

    if model_name == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    return model


def dump_config_model(config_model):
    """ input: a dictionary with the configuration of current run
        process: dump dictionary in a json file in current directory
    """
    json_object = json.dumps(config_model, indent=4)
    with open("model_configuration.json", "w") as outfile:
        outfile.write(json_object)


def plot_figures(history):
    """ input: history
        process:
            - for each metric in history plot the respective diagram
            - save plots in png files in current directory
    """

    plt.figure(figsize=(8, 6))
    plots = [
                {
                    'history': ['train_loss', 'valid_loss'],
                    'ylabel': "Average Negative Log Likelihood",
                    'title': "Training and Validation Losses",
                    'filename': 'loss.png'},
                {
                    'history': ['train_acc', 'valid_acc'],
                    'ylabel': "Average Accuracy",
                    'title': "Training and Validation Accuracy",
                    'filename': 'accuracy.png'},
                {
                    'history': ['train_loss', 'valid_loss'],
                    'ylabel': "Average F1 score macro",
                    'title': "Training and Validation F1 score macro",
                    'filename': 'f1_macro.png'}
            ]

    for i in plots:
        for j in i['history']:
            plt.plot(
                100 * history[j], label=j)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(i['ylabel'])
        plt.title(i['title'])

        plt.savefig(i['filename'])


def main():
    """ 1. Define paths for dataset and parameters related with the model
        2. Call function gather_png to create train / validation / test dataset
        3. Define manual_seed, so as to regenerate the creation of batches
        4. Call function get_pretrained_model, where we initaliaze our model
        5. Call function train_model to train the model and calculate metrics
        6. Finally dump configuration file in a json file and plot all metrics
    """

    # set filename for the model
    save_file_name = 'vgg16-transfer.pth'
    checkpoint_path = 'vgg16-transfer.pth'

    # define input paths for train / test / validation
    train_dir_path = 'dataset/rgb/train'
    val_dir_path = 'dataset/rgb/validation'
    test_dir_path = 'dataset/rgb/test'

    # defining parameters
    batch_size = 64
    max_epochs_stop = 5
    n_epochs = 15
    seed = 10
    lr = 0.01

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data_array, train_label_list = gather_png(train_dir_path)
    val_data_array, val_label_list = gather_png(val_dir_path)
    test_data_array, test_label_list = gather_png(test_dir_path)

    train = TensorDataset(train_data_array.to(device), torch.tensor(train_label_list).long().to(device))
    test = TensorDataset(test_data_array.to(device), torch.tensor(test_label_list).long().to(device))
    val = TensorDataset(val_data_array.to(device), torch.tensor(val_label_list).long().to(device))

    torch.manual_seed(seed)

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(train, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val, batch_size=batch_size, shuffle=True),
        'test': DataLoader(test, batch_size=batch_size, shuffle=True)
    }

    trainiter = iter(dataloaders['train'])
    features, labels = next(trainiter)
    print(features)
    print(features.shape)
    print(labels)
    print(labels.shape)

    # Call pretrained model for vgg and 6 number of classes
    model = get_pretrained_model('vgg16', n_classes=6)
    if torch.cuda.is_available():
        model.cuda()

    print(summary(model, input_size=(3, 224, 224), batch_size=batch_size))

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    # update config_model with related setup (batch_size/epochs/seed/..) and parameters(lr/optimizer/..) used for training the model
    config_model.update({'save_file_name': save_file_name, 'batch_size': batch_size, 'seed': seed, 'max_epochs_stop': max_epochs_stop,\
                         'n_epochs': n_epochs, 'optimizer': 'Adam', 'lr': lr, \
                         'random_state_dataloader': 'default'})

    model, history = train_model(
        model,
        criterion,
        optimizer,
        dataloaders['train'],
        dataloaders['val'],
        save_file_name=save_file_name,
        max_epochs_stop=max_epochs_stop,
        n_epochs=n_epochs,
        print_every=1)

    # dump config_model in configuration json file
    dump_config_model(config_model)
    # plot figures based on history (contains results of accuracy/loss/f1_score)
    plot_figures(history)


if __name__ == "__main__":
    main()