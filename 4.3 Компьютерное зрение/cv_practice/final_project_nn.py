from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import pandas as pd


def load_train(path):
    train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1 / 255.,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2)
    train_datagen_flow = train_datagen.flow_from_dataframe(
        dataframe=pd.read_csv(path + 'labels.csv'),
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        subset='training',
        class_mode='raw',
        seed=12345)
    return train_datagen_flow


def load_test(path):
    val_datagen = ImageDataGenerator(validation_split=0.25, rescale=1 / 255.)
    val_datagen_flow = val_datagen.flow_from_dataframe(
        dataframe=pd.read_csv(path + 'labels.csv'),
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        subset='validation',
        class_mode='raw',
        seed=12345)
    return val_datagen_flow


# ResNet50
# def create_model(input_shape):
#     optimizer = Adam(lr=0.0001)
#     backbone = ResNet50(input_shape=(224, 224, 3),
#                         weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                         include_top=False)
#     model = Sequential()
#     model.add(backbone)
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(units=1, activation='relu'))
#     model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
#     return model
# epochs 6

# VGG19 + Flatten + Dense(1)+Relu
# def create_model(input_shape):
#     optimizer = Adam(lr=0.0001)
#     model = Sequential()
#     vgg = VGG19(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
#     model.add(vgg)
#     model.add(Flatten())
#     model.add(Dense(units=1, activation='relu'))
#     model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
#     return model
# val_mean_absolute_error: 7.3695

# VGG19 + 2 x Dense(4096)+Relu+Dropout
# def create_model(input_shape):
#     optimizer = Adam(lr=0.0001)
#     model = Sequential()
#     vgg = VGG19(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
#     model.add(vgg)
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(units=4096, activation='relu'))
#     model.add(Dropout(rate=0.5, seed=12345))
#     model.add(Dense(units=4096, activation='relu'))
#     model.add(Dropout(rate=0.5, seed=12345))
#     model.add(Dense(units=100, activation='softmax'))
#     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['mean_absolute_error'])
#     return model

def create_model(input_shape):
    optimizer = Adam(lr=0.0001)
    model = Sequential()
    vgg = VGG19(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # vgg.trainable = False
    model.add(vgg)
    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    # model.add(Dense(units=4096, activation='relu'))
    # model.add(Dropout(rate=0.5, seed=12345))
    # model.add(Dense(units=4096, activation='relu'))
    # model.add(Dropout(rate=0.5, seed=12345))
    # model.add(Dense(units=100, activation='relu'))
    # model.add(Dropout(rate=0.5, seed=12345))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=26,
                steps_per_epoch=None, validation_steps=None):
    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    return model
