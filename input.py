import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import math

drive_file = './data/driving_log.csv'
test_size = 0.2
batch_size = 256
adjust_angle = 0.25
n_row = 160
n_col = 320
new_size_col, new_size_row = 64 , 64



def load_split_data():
    '''
    Load driving_log.csv, save to a list and split to train and validation logs
    '''
    with open(drive_file, 'r') as f:
        f.readline()
        data = [line.strip().split(', ') for line in f]
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=2017)
    return train_data, val_data

def data_generator(data):
    '''
    input: list of driving logs
    output: generator of driving logs
    '''
    n_sample = len(data)
    next_epoch = True

    while True:
        if next_epoch:
            indices = np.arange(n_sample)
            np.random.shuffle(indices)
            yield data[indices[0]]
            ind = 1
            next_epoch = False
        else:
            yield data[indices[ind]]
            ind += 1
            if ind >= n_sample:
                next_epoch = True


def train_batch_generator(data, data_augmentation = True):
    '''
    Data preprocessing and augmentation
    input: driving logs
    output: generator of images + steer angles 
    '''
    ind = 0

    for info_frame in data_generator(data):
        if ind == 0:
            batch_sample = np.zeros((batch_size, new_size_col, new_size_row, 3))
            batch_steer = np.zeros((batch_size,))

        # Randomly choose center, left or right frame
        chosen_frame = np.random.randint(3)
        frame = cv2.imread('./data_from_udacity/data/'+ info_frame[chosen_frame])

        if chosen_frame == 2:
            steer = np.float32(info_frame[3]) - adjust_angle
        else:
            steer = np.float32(info_frame[3]) + adjust_angle * chosen_frame
        if data_augmentation:
            # Randomly flip the frame
            flip = np.random.randint(1)
            if flip == 1:
                frame = frame[:, ::-1, :]
                steer *= -1

            # Randomly adjust brightness
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            random_bright = .25 + np.random.uniform()
            frame[:, :, 2] = frame[:, :, 2] * random_bright
            frame = np.clip(frame, 0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)

            # Horizontal and vertical shifts
            horizontal_shift = np.random.randint(-25, 26)
            vertical_shift = np.random.randint(-10,11)

            rows, cols = frame.shape[:2]
            M = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
            frame = cv2.warpAffine(frame, M, (cols, rows))
            steer += 0.035 * horizontal_shift

        # Crop the frame
        #frame = frame[math.floor(n_row / 5):n_row - 25, 0:n_col]
        frame = frame[math.floor(n_row / 5):n_row - 25, 25:n_col - 25]

        frame = cv2.resize(frame, (new_size_col, new_size_row), interpolation=cv2.INTER_AREA)

        batch_sample[ind] = frame
        batch_steer[ind] = steer

        ind += 1

        if ind == batch_size:
            yield batch_sample, batch_steer
            ind = 0

