import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from pygame import mixer
import os



#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
#model_type = "dpt_swin2_tiny_256"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
cap = cv2.VideoCapture(1)
plt.ion()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()


midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

mixer.init()



def longestStringOfOnes(binary_string):
    max_length = 0       # Length of the longest sequence found
    max_start_index = -1 # Starting index of the longest sequence
    current_length = 0   # Length of the current sequence
    current_start = 0    # Starting index of the current sequence

    for index, char in enumerate(binary_string):
        if char == '1':
            if current_length == 0:
                current_start = index
            current_length += 1

            # Update max values if current sequence is longer
            if current_length > max_length:
                max_length = current_length
                max_start_index = current_start
        else:
            current_length = 0  # Reset current length if '0' is encountered

    return max_length, max_start_index

current_state = "clear"


while True:
    start = time.time()

    ret, frame = cap.read()
    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)


    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    plt.clf()
    graph = plt.imshow(output)
    #plt.show()
    # print(output[119][159], output[119][319], output[119][479])
    # print(output[239][159], output[239][319], output[239][479])
    # print(output[359][159], output[359][319], output[359][479])

    colavgs = np.mean(output[350:], axis=0)
    rowavgs = np.mean(output[:, 300:340], axis=1)
    
    threshhold = 700

    colstring = ("".join((list([str(int(i>threshhold)) for i in colavgs]))))
    rowstring = ("".join((list([str(int(i>threshhold)) for i in rowavgs]))))
    widthD = longestStringOfOnes(colstring)
    heightD = longestStringOfOnes(rowstring)

    Owidth = f"widest {widthD[0]} starting at {widthD[1]}"
    Oheight = f"tallest {heightD[0]} starting at {heightD[1]}"

    print(Owidth)
    print(Oheight)


    if widthD[0] > 200:
        if current_state == "clear":
            current_state = "ob"
            if widthD[1]+widthD[0]/2 < 320:
                mixer.music.load('right.mp3')
                mixer.music.play()
            if widthD[1]+widthD[0]/2 >= 320:
                mixer.music.load('left.mp3')
                mixer.music.play()
    if widthD[0] < 190:
        if current_state == "ob":
            current_state = "clear"
            print("clear")

    if heightD[0] > 70 and heightD[0] < 100:
        if heightD[1] > 300:
            mixer.music.load('stairs.mp3')
            mixer.music.play()
            print("Stairs") 


    print()
    end = time.time()
    plt.pause(0.1)
    #print(end-start)