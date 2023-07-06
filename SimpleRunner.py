import os
import numpy as np
import cv2
from MyCNN_batch import Net
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
import mediapipe as mp
from keras.models import load_model
import time
import pandas as pd
from torch.autograd import Variable
from torch.nn import (
    Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module,
    BatchNorm2d, Dropout
)
import torch.nn.functional as F
from torch.optim import Adam
<<<<<<< HEAD
import pandas as pd
import numpy as np
=======
>>>>>>> refs/remotes/origin/Nikolas_branch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import csv
from sklearn import preprocessing
import glob


def main():
<<<<<<< HEAD
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load('Saved_models/4_CNN_1-16-32-6_lr-0.001_batch-1024_n_epochs-100_dropout-0/CNN_1-16-32-6_lr-0.001_batch-1024_n_epochs-100_dropout-0.pth'))
    model.eval()
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape
=======
    """ 1. initialize parameters and define which letters should recognize the model
        2. load cnn model, which is located in directory saved_models
        3. start video capturing using cv2 library
        4. press space to capture a photo ->
           transform input photo (resize / crop / ..)
        5. use the above output as input to our model and predict the letter
        6. the letter will appear in terminal
        7. this process (4-6) is repeated until user presses esc button
    """

    # initialize parameters
>>>>>>> refs/remotes/origin/Nikolas_branch
    analysisframe = ''
    letter_rgb_l = []
    letter_gray_l = []
    pixels_l = []
    le = preprocessing.LabelEncoder()
<<<<<<< HEAD
    letters=['Gamma', 'Beta', 'Eta', 'Phi', 'Theta', 'Zeta']
=======
    letters = ['Gamma', 'Beta', 'Eta', 'Phi', 'Theta', 'Zeta']
>>>>>>> refs/remotes/origin/Nikolas_branch
    le.fit(letters)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load('Saved_models/4_CNN_1-16-32-6_lr-0.001_batch-1024_n_epochs-100_dropout-0/CNN_1-16-32-6_lr-0.001_batch-1024_n_epochs-100_dropout-0.pth'))
    model.eval()
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape

    while True:
        _, frame = cap.read()

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            analysisframe = frame
            showframe = analysisframe
            cv2.imshow("Frame", showframe)
            framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
            analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
            analysisframe = cv2.resize(analysisframe,(28,28))
            
            analysisframe = torch.from_numpy(analysisframe.reshape(1, 1, 28, 28)).to(device)
            analysisframe = analysisframe.float()
            device = next(model.parameters()).device
            with torch.no_grad():
                output = model(analysisframe.to(device))
<<<<<<< HEAD
                
            softmax = F.softmax(output, dim=1)
            top_two_values, top_two_indices = torch.topk(softmax, k=2, dim=1)

            # Convert tensors to lists
            top_two_values = top_two_values.tolist()[0]
            top_two_indices = top_two_indices.tolist()[0]

            # Retrieve probabilities for the top two predictions
            top_two_probabilities = [softmax[0, index].item() for index in top_two_indices]

            print("\n Top two predictions:")
            for rank, (value, index, probability) in enumerate(zip(top_two_values, top_two_indices, top_two_probabilities)):
                if rank+1==2:
                    indentation = "\t"
                else:
                    indentation = ""
                    
                print(f"{indentation} {rank+1}.Prediction: {le.inverse_transform([index])}, Probability: {probability:.2f}")
=======
            softmax = torch.exp(output)
            predictions = torch.argmax(softmax, -1)
            letter = le.inverse_transform([predictions.item()])
            print(letter)
>>>>>>> refs/remotes/origin/Nikolas_branch

            

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
