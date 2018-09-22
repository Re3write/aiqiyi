"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from keras.applications.resnet50 import ResNet50
import numpy as np
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import os.path


# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'fpn.{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(512, 512),
        batch_size=8,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data', 'test'),
        target_size=(512, 512),
        batch_size=8,
        class_mode='categorical')

    return train_generator, validation_generator

def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = ResNet50(weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(574, activation='softmax')(x)

    # this is the model we will mechanicalmeter
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def freeze_all_but_top(model):
    """Used to mechanicalmeter just the top layers of the model."""
    # first: mechanicalmeter only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, mechanicalmeter deeper."""
    # we chose to mechanicalmeter the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=nb_epoch,
        validation_steps=200,
        callbacks=callbacks,
        shuffle=True)
    return model

def main(weights_file):
    model = get_model()
    generators = get_generators()

    # Get and mechanicalmeter the mid layers.
    model = freeze_all_but_mid_and_top(model)
    model = train_model(model, 1000, generators,
                        [checkpointer, early_stopper, tensorboard])


if __name__ == '__main__':
    weights_file = None
    main(weights_file)
