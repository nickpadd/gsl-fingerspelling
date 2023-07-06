import os
import sys
import cv2
from torchvision.models import VGG16_Weights
import torchvision.models as models
from PIL import Image
import csv
from torchvision import transforms
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
from keras.models import load_model
import numpy as np
import time
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import (
    Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax,
    BatchNorm2d, Dropout
)
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
import glob


def main():
    """ 1. initialize parameters and define which letters should recognize the model
        2. load vgg16 model, which is located in current directory
        3. start video capturing using cv2 library
        4. press space to capture a photo ->
           transform input photo (resize / crop / ..)
        5. use the above output as input to our model and predict the letter
        6. the letter will appear in terminal
        7. this process (4-6) is repeated until user presses esc button
    """

    # initialize parameters
    analysisframe = ''
    letter_rgb_l = []
    letter_gray_l = []
    pixels_l = []
    le = preprocessing.LabelEncoder()

    letters=['Gamma', 'Beta', 'Eta', 'Phi', 'Theta', 'Zeta']
    le.fit(letters)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('vgg16-transfer-final.pth', map_location=torch.device('cpu'))
    model.eval()

    cap = cv2.VideoCapture(0)
    _, frame = cap.read()

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
            cv2.imshow("Frame", analysisframe)
            analysisframe = Image.fromarray(analysisframe)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            analysisframe = preprocess(analysisframe)
            analysisframe = analysisframe.unsqueeze(0).to(device)
            device = next(model.parameters()).device
            with torch.no_grad():
                output = model(analysisframe.to(device))

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

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
