from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Dropout, ELU, Lambda
from keras.callbacks import ModelCheckpoint, CSVLogger
from input import *


def model(summary = True):
    