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
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import os.path
from keras.applications.imagenet_utils import _obtain_input_shape

############################################################
#  Utility Functions
############################################################
# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'fpn.{epoch:03d}-{val_loss:.2f}.hdf5'),
    monitor='val_loss',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))



# class BatchNorm(KL.BatchNormalization):
#     """Batch Normalization class. Subclasses the Keras BN class and
#     hardcodes training=False so the BN layer doesn't update
#     during training.
#     Batch normalization has a negative effect on training if batches are small
#     so we disable it here.
#     """
#
#     def call(self, inputs, training=None):
#         return super(self.__class__, self).call(inputs, training=False)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = KL.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = KL.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def build(input_tensor=None,input_shape=None):
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=216,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=None)

    if input_tensor is None:
        img_input = KL.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = KL.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

        # Image size must be dividable by 2 multiple times
 # "Image size must be dividable by 2 at least 6 times "
 #                            "to avoid fractions when downscaling and upscaling."
 #                            "For example, use 256, 320, 384, 448, 512, ... etc. "

        # Inputs

    _, C2, C3, C4, C5 = resnet_graph(img_input, "resnet50", stage5=True)
    # Top-down Layers
    # TODO: add assert to varify feature map sizes match what's in config
    P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)

    mrcnn_feature_maps = [P2, P3, P4, P5]

    # Generate Anchors
    model = KM.Model(img_input, mrcnn_feature_maps, name='fpn')

    return model

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

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(216, 216),
        batch_size=16,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data', 'test'),
        target_size=(216, 216),
        batch_size=32,
        class_mode='categorical')

    return train_generator,validation_generator

def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = build()
    # add a global spatial average pooling layer
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    # prediction0 = Dense(574, activation='softmax')(x)

    x0=GlobalAveragePooling2D()(x[0])
    x0 = Dense(1024, activation='relu')(x0)
    prediction0 = Dense(574, activation='softmax')(x0)

    x1=GlobalAveragePooling2D()(x[1])
    x1 = Dense(1024, activation='relu')(x1)
    prediction1 = Dense(574, activation='softmax')(x1)

    x2=GlobalAveragePooling2D()(x[2])
    x2 = Dense(1024, activation='relu')(x2)
    prediction2 = Dense(574, activation='softmax')(x2)

    x3=GlobalAveragePooling2D()(x[3])
    x3 = Dense(1024, activation='relu')(x3)
    prediction3 = Dense(574, activation='softmax')(x3)


    predictions=KL.add([prediction0,prediction1,prediction2,prediction3])

    # this is the model we will mechanicalmeter
    model = Model(inputs=base_model.input, outputs=predictions)
    return model



def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator ,val_generator= generators
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=200,
        epochs=nb_epoch,
        callbacks=callbacks,
        shuffle=True)
    return model

def main(weights_file):
    model=get_model()
    #model = get_model()
    generators = get_generators()
    model.compile(
        #optimizer=SGD(lr=0.0001, momentum=0.9),
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    model = train_model(model, 1000, generators,
                      [checkpointer,early_stopper,tensorboard])
if __name__ == '__main__':
    weights_file = None
    main(weights_file)
