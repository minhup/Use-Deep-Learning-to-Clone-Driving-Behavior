from keras.models import Model
from keras.layers import Input, Activation, merge
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Dense, merge
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.regularizers import l2
from input import *



def model(summary = True):
    
    input_img = Input((64, 64, 3))

    normalized_input = Lambda(lambda z: z / 255. - .5)(input_img)


    conv = Convolution2D(8, 3, 3, activation='relu', init='glorot_uniform',
                         border_mode='same')(normalized_input)
    conv1 = Convolution2D(8, 3, 3, activation='relu', init='glorot_uniform',
                          border_mode='same')(normalized_input)
    conv2 = Convolution2D(8, 3, 3, activation='relu', init='glorot_uniform',
                          border_mode='same')(conv1)
    merge1 = merge([conv, conv2], mode='concat', concat_axis=3)
    maxpool = MaxPooling2D((2, 2))(merge1)
    dropout = Dropout(0.5)(maxpool)



    conv = Convolution2D(16, 3, 3, activation='relu', init='glorot_uniform',
                          border_mode='same')(dropout)
    conv1 = Convolution2D(16, 3, 3, activation='relu', init='glorot_uniform',
                          border_mode='same')(dropout)
    conv2 = Convolution2D(16, 3, 3, activation='relu', init='glorot_uniform',
                          border_mode='same')(conv1)
    merge1 = merge([conv, conv2], mode='concat', concat_axis=3)
    maxpool = MaxPooling2D((2, 2))(merge1)
    dropout = Dropout(0.5)(maxpool)


    conv = Convolution2D(32, 3, 3, activation='relu', init='glorot_uniform',
                         border_mode='same')(dropout)
    conv1 = Convolution2D(32, 3, 3, activation='relu', init='glorot_uniform',
                          border_mode='same')(dropout)
    conv2 = Convolution2D(32, 3, 3, activation='relu', init='glorot_uniform',
                          border_mode='same')(conv1)
    merge1 = merge([conv, conv2], mode='concat', concat_axis=3)
    maxpool = MaxPooling2D((2, 2))(merge1)
    dropout = Dropout(0.5)(maxpool)


    flatten = Flatten()(dropout)

    dense = Dense(64, activation='relu', init='glorot_uniform', W_regularizer=l2(0.01) )(flatten)
    dropout = Dropout(0.5)(dense)

    out = Dense(1)(dropout)
    #out = Dense(1)(flatten)

    model = Model(input=input_img, output=out)

    if summary:
        model.summary()

    return model

if __name__ == '__main__':

    train_data, val_data = load_split_data()

    model_ = model(summary=True)

    opt = Adam(lr=5e-4)
    model_.compile(optimizer=opt, loss='mse')

    # json dump of model architecture
    with open('checkpoints/model.json', 'w') as f:
        f.write(model_.to_json())

    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.h5')
    logger = CSVLogger(filename='checkpoints/history.csv')

    model_.fit_generator(generator=train_batch_generator(train_data, data_augmentation = True),
                            samples_per_epoch=100*batch_size,
                            nb_epoch=100,
                            validation_data=train_batch_generator(val_data, data_augmentation = False),
                            nb_val_samples=20*batch_size,
                            callbacks=[checkpointer, logger])

