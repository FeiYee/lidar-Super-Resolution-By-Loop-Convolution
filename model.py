import os
import datetime

from data import *

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.layers import *

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
# K.set_floatx('float16')
K.set_floatx('float32')
# print(K.floatx())
###########################################################################################
#################################   Model Specification   #################################
###########################################################################################

# Create Model 
model_name = 'UNet'
# case_name = model_name + '-mae-' + data_set + '-from-' + str(image_rows_low) + '-to-' + str(image_rows_high)
case_name = "your_prediction"

# automatically generate log and weight path
log_path = os.path.join(root_dir, 'logs', case_name)
weight_path = os.path.join(root_dir, 'weights', case_name)
weight_name = os.path.join(weight_path, 'weights.h5')
print('#'*50)
print('Using model:              {}'.format(model_name))
print('Trainig case:             {}'.format(case_name))
print('Log directory:            {}'.format(log_path))
print('Weight directory:         {}'.format(weight_path))
print('Weight name:              {}'.format(weight_name))
path_lists = [log_path, weight_path]
for folder_name in path_lists:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)

def expend_as(tensor, rep):

    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                       arguments={'repnum': rep})(tensor)
    return my_repeat

def AttnGatingBlock(x, g, inter_shape,name = ""):

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(filters=inter_shape,
                   kernel_size=1,
                   strides=1,
                   padding='same',name = name + "_2dConv1")(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(filters=inter_shape,
                     kernel_size=3,
                     strides=(shape_x[1] // shape_g[1],
                              shape_x[2] // shape_g[2]),
                     padding='same',name = name + "_2dConv2")(x)

    # Element-wise addition of the gating and x signals
    add_xg = add([phi_g, theta_x])
    add_xg = Activation('relu',name = name + "_2drelu")(add_xg)

    # 1x1x1 convolution
    psi = Conv2D(filters=1, kernel_size=1, padding='same',name = name + "_2dConv3")(add_xg)
    psi = Activation('sigmoid',name = name + "_2dsigmoid")(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1],
                                             shape_x[2] //
                                             shape_sigmoid[2]),name = name + "_2dUP")(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[3])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv2D(filters=inter_shape,
                    kernel_size=1,
                    strides=1,
                    padding='same',name = name + "_2dConv4")(attn_coefficients)
    output = BatchNormalization(name = name + "_bn")(output)
    return output









###########################################################################################
#################################        model        #####################################
###########################################################################################

def UNet():

    def up_block(input, filters=64, kernel_size=(3,3), strides=(1,1)):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer=kernel_init,name = "conv2d_transpose")(input)
        x = BatchNormalization(name = "batch_normalization_v1")(x)
        x = Activation(act_func)(x)
        return x

    def up_block2(input, filters=64, kernel_size=(3,3), strides=(1,1)):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer=kernel_init,name = "conv2d_transpose_1")(input)
        x = BatchNormalization(name = "batch_normalization_v1_1")(x)
        x = Activation(act_func)(x)
        return x

    filters = 64
    dropout_rate = 0.25
    act_func = 'relu'
    kernel_init = 'he_normal'


    encoder_block1_1 = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d")
    encoder_block1_2 = BatchNormalization(name = "batch_normalization_v1_2")
    encoder_block1_3 = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_1")
    encoder_block1_4 = BatchNormalization(name = "batch_normalization_v1_3")


    
    encoder_block2_1 = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_2")
    encoder_block2_2 = BatchNormalization(name = "batch_normalization_v1_4")
    encoder_block2_3 = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_3")
    encoder_block2_4 = BatchNormalization(name = "batch_normalization_v1_5")


    
    
    encoder_block3_1 = Conv2D(filters=filters*4, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_4")
    encoder_block3_2 = BatchNormalization(name = "batch_normalization_v1_6")
    encoder_block3_3 = Conv2D(filters=filters*4, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_5")
    encoder_block3_4 = BatchNormalization(name = "batch_normalization_v1_7")

    
    
    encoder_block4_1 = Conv2D(filters=filters*8, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_6")
    encoder_block4_2 = BatchNormalization(name = "batch_normalization_v1_8")
    encoder_block4_3 = Conv2D(filters=filters*8, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "2conv2d_7")
    encoder_block4_4 = BatchNormalization(name = "batch_normalization_v1_9")


    
    encoder_block44_1 = Conv2D(filters=filters*16, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_8")
    encoder_block44_2 = BatchNormalization(name = "batch_normalization_v1_10")
    encoder_block44_3 = Conv2D(filters=filters*16, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_9")
    encoder_block44_4 = BatchNormalization(name = "batch_normalization_v1_11")


    
    encoder_block33_1 = Conv2D(filters=filters*8, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_10")
    encoder_block33_2 = BatchNormalization(name = "batch_normalization_v1_13")
    encoder_block33_3 = Conv2D(filters=filters*8, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_11")
    encoder_block33_4 = BatchNormalization(name = "batch_normalization_v1_14")


    
    encoder_block22_1 = Conv2D(filters=filters*4, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_12")
    encoder_block22_2 = BatchNormalization(name = "batch_normalization_v1_16")
    encoder_block22_3 = Conv2D(filters=filters*4, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_13")
    encoder_block22_4 = BatchNormalization(name = "batch_normalization_v1_17")



    encoder_block11_1 = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_14")
    encoder_block11_2 = BatchNormalization(name = "batch_normalization_v1_19")
    encoder_block11_3 = Conv2D(filters=filters*2, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_15")
    encoder_block11_4 = BatchNormalization(name = "batch_normalization_v1_20")


    
    
    encoder_block0_1 = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_16")
    encoder_block0_2 = BatchNormalization(name = "batch_normalization_v1_22")
    encoder_block0_3 = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer=kernel_init,name = "conv2d_17")
    encoder_block0_4 = BatchNormalization(name = "batch_normalization_v1_23")




    decoder_block4_1 = Conv2DTranspose(filters=filters * 8, kernel_size=(3,3), padding='same', strides=(2,2), kernel_initializer=kernel_init,name = "conv2d_transpose_2")
    decoder_block4_2 = BatchNormalization(name = "batch_normalization_v1_12")


    
    decoder_block3_1 = Conv2DTranspose(filters=filters * 4, kernel_size=(3,3), padding='same', strides=(2,2), kernel_initializer=kernel_init,name = "conv2d_transpose_3")
    decoder_block3_2 = BatchNormalization(name = "batch_normalization_v1_15")


    
    decoder_block2_1 = Conv2DTranspose(filters=filters * 2, kernel_size=(3,3), padding='same', strides=(2,2), kernel_initializer=kernel_init,name = "conv2d_transpose_4")
    decoder_block2_2 = BatchNormalization(name = "batch_normalization_v1_18")


    
    decoder_block1_1 = Conv2DTranspose(filters=filters, kernel_size=(3,3), padding='same', strides=(2,2), kernel_initializer=kernel_init,name = "conv2d_transpose_5")
    decoder_block1_2 = BatchNormalization(name = "batch_normalization_v1_21")
    inputs = Input((image_rows_low, image_cols, channel_num))

    # upscailing
    x0 = inputs
    x0 = up_block(x0, filters, strides=(2,1))
    x0 = up_block2(x0, filters, strides=(2,1))

    x1 = encoder_block1_1(x0)
    x1 = encoder_block1_2(x1)
    x1 = Activation(act_func)(x1)
    x1 = encoder_block1_3(x1)
    x1 = encoder_block1_4(x1)
    x1 = Activation(act_func)(x1)


    x2 = AveragePooling2D((2,2))(x1)
    x2 = Dropout(dropout_rate)(x2, training=True)
    x2 = encoder_block2_1(x2)
    x2 = encoder_block2_2(x2)
    x2 = Activation(act_func)(x2)
    x2 = encoder_block2_3(x2)
    x2 = encoder_block2_4(x2)
    x2 = Activation(act_func)(x2)
    

    
    x3 = AveragePooling2D((2,2))(x2)
    x3 = Dropout(dropout_rate)(x3, training=True)
    x3 = encoder_block3_1(x3)
    x3 = encoder_block3_2(x3)
    x3 = Activation(act_func)(x3)
    x3 = encoder_block3_3(x3)
    x3 = encoder_block3_4(x3)
    x3 = Activation(act_func)(x3)
     

    x4 = AveragePooling2D((2,2))(x3)
    x4 = Dropout(dropout_rate)(x4, training=True)
    x4 = encoder_block4_1(x4)
    x4 = encoder_block4_2(x4)
    x4 = Activation(act_func)(x4)
    x4 = encoder_block4_3(x4)
    x4 = encoder_block4_4(x4)
    x4 = Activation(act_func)(x4)


     
    y4 = AveragePooling2D((2,2))(x4)
    y4 = Dropout(dropout_rate)(y4, training=True)
    y4 = encoder_block44_1(y4)
    y4 = encoder_block44_2(y4)
    y4 = Activation(act_func)(y4)
    y4 = encoder_block44_3(y4)
    y4 = encoder_block44_4(y4)
    y4 = Activation(act_func)(y4)
    y4 = Dropout(dropout_rate)(y4, training=True)
    y4 = decoder_block4_1(y4)
    y4 = decoder_block4_2(y4)


    y3 = concatenate([x4, y4], axis=3)
    y3 = encoder_block33_1(y3)
    y3 = encoder_block33_2(y3)
    y3 = Activation(act_func)(y3)
    y3 = encoder_block33_3(y3)
    y3 = encoder_block33_4(y3)
    y3 = Activation(act_func)(y3)
    y3 = Dropout(dropout_rate)(y3, training=True)
    y3 = decoder_block3_1(y3)
    y3 = decoder_block3_2(y3)

    y2 = concatenate([x3, y3], axis=3)
    y2 = encoder_block22_1(y2)
    y2 = encoder_block22_2(y2)
    y2 = Activation(act_func)(y2)
    y2 = encoder_block22_3(y2)
    y2 = encoder_block22_4(y2)
    y2 = Activation(act_func)(y2)
    y2 = Dropout(dropout_rate)(y2, training=True)
    y2 = decoder_block2_1(y2)
    y2 = decoder_block2_2(y2)
    ''''''


    '''
    第二次递归
    '''
    attn = AttnGatingBlock(x2,y2,filters * 2,"attn_2_1")
    x_3 = AveragePooling2D((2,2))(attn)
    x_3 = Dropout(dropout_rate)(x_3, training=True)
    x_3 = encoder_block3_1(x_3)
    x_3 = encoder_block3_2(x_3)
    x_3 = Activation(act_func)(x_3)
    x_3 = encoder_block3_3(x_3)
    x_3 = encoder_block3_4(x_3)
    x_3 = Activation(act_func)(x_3)


    attn = AttnGatingBlock(x_3, y3, filters * 4,"attn_2_2")
    x_4_2 = AveragePooling2D((2,2))(attn)
    x_4_2 = Dropout(dropout_rate)(x_4_2, training=True)
    x_4_2 = encoder_block4_1(x_4_2)
    x_4_2 = encoder_block4_2(x_4_2)
    x_4_2 = Activation(act_func)(x_4_2)
    x_4_2 = encoder_block4_3(x_4_2)
    x_4_2 = encoder_block4_4(x_4_2)
    x_4_2 = Activation(act_func)(x_4_2)

    attn = AttnGatingBlock(x_4_2,x4,filters * 8,"attn_2_3")
    y_4_2 = AveragePooling2D((2,2))(attn)
    y_4_2 = Dropout(dropout_rate)(y_4_2, training=True)
    y_4_2 = encoder_block44_1(y_4_2)
    y_4_2 = encoder_block44_2(y_4_2)
    y_4_2 = Activation(act_func)(y_4_2)
    y_4_2 = encoder_block44_3(y_4_2)
    y_4_2 = encoder_block44_4(y_4_2)
    y_4_2 = Activation(act_func)(y_4_2)
    y_4_2 = Dropout(dropout_rate)(y_4_2, training=True)
    y_4_2 = decoder_block4_1(y_4_2)
    y_4_2 = decoder_block4_2(y_4_2)

    
    attn = AttnGatingBlock(x_4_2,y4, filters * 8,"attn_2_4")
    y_3_2 = concatenate([attn, y_4_2], axis=3)
    y_3_2 = encoder_block33_1(y_3_2)
    y_3_2 = encoder_block33_2(y_3_2)
    y_3_2 = Activation(act_func)(y_3_2)
    y_3_2 = encoder_block33_3(y_3_2)
    y_3_2 = encoder_block33_4(y_3_2)
    y_3_2 = Activation(act_func)(y_3_2)
    y_3_2 = Dropout(dropout_rate)(y_3_2, training=True)
    y_3_2 = decoder_block3_1(y_3_2)
    y_3_2 = decoder_block3_2(y_3_2)
    
    
    attn = AttnGatingBlock(concatenate([x_3,x3], axis=3),
                           concatenate([y_3_2, y3], axis=3),
                           filters * 4,"attn_2_5")
    x_4_3 = AveragePooling2D((2,2))(attn)
    x_4_3 = Dropout(dropout_rate)(x_4_3, training=True)
    x_4_3 = encoder_block4_1(x_4_3)
    x_4_3 = encoder_block4_2(x_4_3)
    x_4_3 = Activation(act_func)(x_4_3)
    x_4_3 = encoder_block4_3(x_4_3)
    x_4_3 = encoder_block4_4(x_4_3)
    x_4_3 = Activation(act_func)(x_4_3)

    
    attn = AttnGatingBlock(x_4_3,x_4_2,filters * 8,"attn_2_6")
    y_4_3 = AveragePooling2D((2,2))(attn)
    y_4_3 = Dropout(dropout_rate)(y_4_3, training=True)
    y_4_3 = encoder_block44_1(y_4_3)
    y_4_3 = encoder_block44_2(y_4_3)
    y_4_3 = Activation(act_func)(y_4_3)
    y_4_3 = encoder_block44_3(y_4_3)
    y_4_3 = encoder_block44_4(y_4_3)
    y_4_3 = Activation(act_func)(y_4_3)
    y_4_3 = Dropout(dropout_rate)(y_4_3, training=True)
    y_4_3 = decoder_block4_1(y_4_3)
    y_4_3 = decoder_block4_2(y_4_3)

    

    attn = AttnGatingBlock(concatenate([x_4_3,x_4_2], axis=3),
                           concatenate([y_4_3, y_4_2], axis=3),
                           filters * 16,"attn_2_7x16")
    y_3_3 = encoder_block33_1(attn)
    y_3_3 = encoder_block33_2(y_3_3)
    y_3_3 = Activation(act_func)(y_3_3)
    y_3_3 = encoder_block33_3(y_3_3)
    y_3_3 = encoder_block33_4(y_3_3)
    y_3_3 = Activation(act_func)(y_3_3)
    y_3_3 = Dropout(dropout_rate)(y_3_3, training=True)
    y_3_3 = decoder_block3_1(y_3_3)
    y_3_3 = decoder_block3_2(y_3_3)
 
    
    
    attn = AttnGatingBlock(concatenate([x_3,x3], axis=3),
                           concatenate([y_3_3, y_3_2], axis=3),
                           filters * 8,"attn_2_8")
    y_2 = encoder_block22_1(attn)
    y_2 = encoder_block22_2(y_2)
    y_2 = Activation(act_func)(y_2)
    y_2 = encoder_block22_3(y_2)
    y_2 = encoder_block22_4(y_2)
    y_2 = Activation(act_func)(y_2)
    y_2 = Dropout(dropout_rate)(y_2, training=True)
    y_2 = decoder_block2_1(y_2)
    y_2 = decoder_block2_2(y_2)
    
    
    attn = AttnGatingBlock(y_2,y2,filters * 2,"attn_2_9")
    y1 = concatenate([x2, attn], axis=3)
    y1 = encoder_block11_1(y1)
    y1 = encoder_block11_2(y1)
    y1 = Activation(act_func)(y1)
    y1 = encoder_block11_3(y1)
    y1 = encoder_block11_4(y1)
    y1 = Activation(act_func)(y1)
    y1 = Dropout(dropout_rate)(y1, training=True)
    y1 = decoder_block1_1(y1)
    y1 = decoder_block1_2(y1)
#     ''''''

 


    y0 = concatenate([x1, y1], axis=3)
    y0 = encoder_block0_1(y0)
    y0 = encoder_block0_2(y0)
    y0 = Activation(act_func)(y0)
    y0 = encoder_block0_3(y0)
    y0 = encoder_block0_4(y0)
    y0 = Activation(act_func)(y0)

    outputs = Conv2D(1, (1, 1), activation=act_func,name="conv2d_18")(y0)

    model = Model(inputs=inputs, outputs=outputs)

#     model.compile(
#         optimizer=Adam(lr=0.0001, decay=0.00001),
#         loss='mae'
#     )
    model.compile(
        optimizer=SGD(lr=0.0001, decay=0.00001),
        loss='mae',
#         metrics=['mse']
    )
    model.summary()

    return model



###########################################################################################
#################################   some functions    #####################################
###########################################################################################

def create_case_dir(type_name):
    # tensorboard
    model_checkpoint = None
    tensorboard = None
    os.system('killall tensorboard')
    # create tensorboard checkpoint
    if type_name == 'training':
        model_checkpoint = ModelCheckpoint(weight_name, save_best_only=True, period=1)
        tensorboard = TensorBoard(log_dir=log_path)
        # run tensorboard
        command = 'tensorboard --logdir=' + os.path.join(root_dir, 'logs') + ' &'
        os.system(command)
        # delete old log files
        for the_file in os.listdir(log_path):
            file_path = os.path.join(log_path, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    return model_checkpoint, tensorboard


def get_model(type_name='training'):
    # create case dir
    model_checkpoint, tensorboard = create_case_dir(type_name)
    # create default model
    model = None
    # Choose Model
    if model_name == 'UNet':
        model = UNet()

    return model, model_checkpoint, tensorboard
