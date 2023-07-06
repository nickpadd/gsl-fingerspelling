import os
import numpy as np
import sys
import cv2
from PIL import Image
import csv


if not os.path.exists('rgb'):
    os.mkdir('rgb')

if not os.path.exists('gray'):
    os.mkdir('gray')


def save_png(letter, letter_gray_l, letter_rgb_l):
    counter = 0
    for row in letter_gray_l:
        image = Image.fromarray(row)
        counter += 1

        filename = 'gray/{}{}.png'.format(letter, counter)
        image.save(filename)
        print('saved:', filename)

    counter = 0
    for row in letter_rgb_l:
        image = Image.fromarray(row)
        counter += 1

        filename = 'rgb/{}{}.png'.format(letter, counter)
        image.save(filename)
        print('saved:', filename)


def save_pixels(letter, letter_l):
    counter = 0
    pixels = []
    with open(letter+".csv", "w") as fp:
        write = csv.writer(fp)
        for items in letter_l:
            for item in items:
                for i in item:
                    pixels.append(i)
            write.writerow(pixels)
            pixels = []

def main():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape
    analysisframe = ''
    letter_rgb_l = []
    letter_gray_l = []
    pixels_l = []
    letter = sys.argv[1]

    while True:
        _, frame = cap.read()

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            save_pixels(letter, pixels_l)
            save_png(letter, letter_gray_l, letter_rgb_l)
            break
        elif k%256 == 32:
            # SPACE pressed
            analysisframe = frame
            showframe = analysisframe
            cv2.imshow("Frame", showframe)
            framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
            analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
            analysisframe = cv2.resize(analysisframe,(28,28))
            letter_gray_l.append(analysisframe)
            letter_rgb_l.append(framergbanalysis)
            pixels_l.append(analysisframe.tolist())
            print('Frame captured')

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
