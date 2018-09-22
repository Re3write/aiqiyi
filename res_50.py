import os
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import cv2
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping,ReduceLROnPlateau,CSVLogger
import os.path
from keras.applications.imagenet_utils import _obtain_input_shape
from se_block import squeeze_excite_block
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from vlad_pooling import VLAD
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

############################################################
#  Utility Functions
############################################################
# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'Afterwashpart3+vlad.{epoch:03d}-{val_loss:.2f}.hdf5'),
    monitor='val_acc',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

csv_log=CSVLogger("result.csv")


LrReduce= ReduceLROnPlateau(factor=0.5,patience=2,min_lr=0.000001)



def find_last(self):
        """Finds the last checkpoints file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoints file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoints
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])

        return dir_name, checkpoint




def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        # horizontal_flip=True,
        # rotation_range=10.,
        # width_shift_range=0.2,
        # height_shift_range=0.2
        )

    test_datagen = ImageDataGenerator(rescale=1./255,
                     )

    train_generator = train_datagen.flow_from_directory(
        '/home/sk49/workspace/dataset/QIYI_FACE/video/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        '/home/sk49/workspace/dataset/QIYI_FACE/video/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    return train_generator,validation_generator

def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionResNetV2(weights=weights, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    x =squeeze_excite_block(x)
    x = GlobalAveragePooling2D()(x)
    # x = VLAD(64)(x)
    # x = Dropout(0.5)(x)
    # let's add a fully-connected layer
    # x = Dense(5012, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(4934, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model



def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator ,val_generator= generators
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        epochs=nb_epoch,
        callbacks=callbacks,
        shuffle=True)
    return model

def main(weights_file):
    model=get_model()
    #model = get_model()
    generators = get_generators()
    model.compile(
        optimizer=SGD(lr=0.001, momentum=0.9),
        #optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    model = train_model(model, 200, generators,
                      [checkpointer,early_stopper,tensorboard,LrReduce,csv_log])
if __name__ == '__main__':
    weights_file = None
    main(weights_file)
