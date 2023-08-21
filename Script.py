import time
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
import HandTrackingModule as htm
from pynput.mouse import Controller, Button

model = tf.saved_model.load('Keras models v2') # A path to a pre-trained model


def calculate_sensitivity(x, y, prev_x, prev_y, time_interval=1):
    # Calculate the change in x and y coordinates
    delta_x = x - prev_x
    delta_y = y - prev_y

    # Calculate the velocity in x and y directions
    velocity_x = delta_x / time_interval
    velocity_y = delta_y / time_interval

    # Calculate the sensitivity based on the magnitude of velocity
    sensitivity = (abs(velocity_x) + abs(velocity_y)) / 2

    return sensitivity


def calculate_relative_coordinates(lmList):
    reference_landmark_id = 9
    reference_landmark_x, reference_landmark_y = lmList[reference_landmark_id][1:3]
    fifth_landmark_x, fifth_landmark_y = lmList[5][1:3]
    sixth_landmark_x, sixth_landmark_y = lmList[6][1:3]
    length = ((sixth_landmark_x - fifth_landmark_x) ** 2 + (sixth_landmark_y - fifth_landmark_y) ** 2) ** 0.5
    relative_coordinates = []

    for landmark in lmList:
        landmark_id, landmark_x, landmark_y = landmark
        relative_x = round((landmark_x - reference_landmark_x) / length, 4)
        relative_y = round((landmark_y - reference_landmark_y) / length, 4)
        relative_coordinates.append([landmark_id, relative_x, relative_y])

    return relative_coordinates


def prepare_data(lmList):
    data_raw = []
    for line in lmList:
        for i in range(len(line)):
            if line[i] == 9 and i == 0:
                break
            elif i == 0:
                pass
            else:
                data_raw.append(line[i])
    # print(data_raw)
    dataset = tf.convert_to_tensor([data_raw])
    return dataset


def calculate_action(predictions):
    zeros = predictions.count(0)
    ones = predictions.count(1)
    twos = predictions.count(2)
    threes = predictions.count(3)
    if zeros > ones and zeros > twos and zeros > threes:
        return 0
    elif ones > zeros and ones > twos and ones > threes:
        return 1
    elif twos > zeros and twos > ones and twos > threes:
        return 2
    elif threes > zeros and threes > ones and threes > twos:
        return 3
    else:
        return 0


def mid_dot(dots):
    # For mid x, y track only lmarks No. 0, 5, 9, 13, 17
    acceptable = [0, 5, 9, 13, 17]
    x_sum = 0
    y_sum = 0

    for dot in dots:
        if dot[0] in acceptable:
            x, y = dot[1:3]
            x_sum += x
            y_sum += y

    num_dots = len(acceptable)
    middle_x = x_sum // num_dots
    middle_y = y_sum // num_dots

    return middle_x, middle_y

'''
### This part of code uses different approach to handle user clicks ###
def click_handler(action, flag):
    if action == 1:
        if flag == 'flag_r':
            Controller().release(Button.right)
        Controller().press(Button.left)
        flag = 'flag_l'
    elif action == 2:
        if flag == 'flag_l':
            Controller().release(Button.left)
        Controller().press(Button.right)
        flag = 'flag_r'
    elif action == 0:
        if flag == 'flag_r':
            Controller().release(Button.right)
        if flag == 'flag_l':
            Controller().release(Button.left)
        flag = None
    elif action == 3:
        if flag == 'flag_r':
            Controller().release(Button.right)
        if flag == 'flag_l':
            Controller().release(Button.left)
        flag = None
    return flag
'''

def big_point(lmList, flag):
    x_b = lmList[4][1]
    y_b = lmList[4][2]
    x_p = lmList[8][1]
    y_p = lmList[8][2]
    leng = ((x_b - x_p)**2 + (y_b - y_p)**2)**0.5
    if leng < 20:
        Controller().press(Button.left)
        flag = 'flag_l'
    elif flag == 'flag_l':
        Controller().release(Button.left)
        flag = None
    return flag


def main():
    cap = cv2.VideoCapture(1)
    detector = htm.handDetector(detectionCon=0.7, trackCon=0.7)
    pTime = 0
    predictions = [0] * 15
    sum_x = [0] * 15
    sum_y = [0] * 15
    sum_sens = [0] * 15
    prev_x, prev_y = Controller().position
    flag = 0
    click_flag = None

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        # calculate FPS and show
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        if len(lmList) != 0:
            # making prediction
            crc = calculate_relative_coordinates(lmList)
            dataset = prepare_data(crc)
            prediction = model(dataset).numpy()[0][0]

            # cv2 text
            cv2.putText(img, str(round(prediction)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            click_flag = big_point(lmList, click_flag)

            if prediction > 0.5:
                # find x and y of middle dot
                x, y = mid_dot(lmList)

                # show the middle dot
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

                # understand if it is first time seeing not 3
                if flag == 0:
                    dx = prev_x - x
                    dy = prev_y - y
                else:
                    dx = 0
                    dy = 0
                    flag = 0

                # add x and y
                sum_x.append(dx)
                sum_x.pop(0)
                sum_y.append(dy)
                sum_y.pop(0)

                # calculate movement vector
                Dx, Dy = sum(sum_x) / 15, sum(sum_y) / 15

                # calculate current dots
                cur_x, cur_y = Controller().position

                # move mouse
                Controller().position = (Dx * 5 + cur_x, Dy * 5 + cur_y)

                # previous dot
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = Controller().position
                flag = 1

        # show image
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()
