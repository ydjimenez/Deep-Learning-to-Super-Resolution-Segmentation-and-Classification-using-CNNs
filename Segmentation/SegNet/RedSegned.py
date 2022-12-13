
from keras.models import *
from keras.layers import *

def SegNet(input_altura=128, input_ancho=128, image_ordering="channels_last"):

    assert input_altura%16 == 0
    assert input_ancho%16 == 0
    
    if image_ordering=="channels_first":
        forma_entrada = (3,input_altura,input_ancho)
        channel_axis = 1
    
    else:
        forma_entrada = (input_altura,input_ancho,3)
        channel_axis = 3 
    input_img = Input(shape=forma_entrada)
    
    # Codificador vgg19
    
    # Bloque 1 
    # convolution block 1
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv1_1', data_format=image_ordering)(input_img)
    x = BatchNormalization(axis=channel_axis, name='block1_norm1')(x)
    
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv1_2', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block1_norm2')(x)
    
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool',data_format=image_ordering)(x)
    f1 = x

    # convolution block 2
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv2_1', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block2_norm1')(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv2_2', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block2_norm2')(x)
    
    x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool',data_format=image_ordering)(x)
    f2 = x

    # convolution block 3
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_1', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block3_norm1')(x)
    
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_2', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block3_norm2')(x)
    
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_3', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block3_norm3')(x)
    
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_4', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block3_norm4')(x)
    
    x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool',data_format=image_ordering)(x)
    f3 = x

    # convolution block 4
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_1', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block4_norm1')(x)
    
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_2', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block4_norm2')(x)
    
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_3', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block4_norm3')(x)
    
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_4', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block4_norm4')(x)
    
    x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool',data_format=image_ordering)(x)
    f4 = x

    # convolution block 5
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_1', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block5_norm1')(x)
    
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_2', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block5_norm2')(x)
    
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_3', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block5_norm3')(x)
    
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv5_4', data_format=image_ordering)(x)
    x = BatchNormalization(axis=channel_axis, name='block5_norm4')(x)
    
    x = MaxPooling2D((2,2), strides=(2,2), data_format=image_ordering)(x)
    f5 = x
    
    #Decodificador
    # upsampling block 7
    o = f5
    o = concatenate([o, f5], axis=channel_axis)
    o = UpSampling2D(name='upsampling7', size=(2,2))(o)
    
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='conv7_1', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block7_norm1')(o)
    
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='conv7_2', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block7_nor2')(o)
    
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='conv7_3', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block7_norm3')(o)
    
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='conv7_4', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block7_norm4')(o)

    # upsampling block 8
    o = concatenate([o, f4], axis=channel_axis)
    o = UpSampling2D(name='upsampling8', size=(2,2))(o)
    
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='conv8_1', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block8_norm1')(o)
    
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='conv8_2', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block8_norm2')(o)
    
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='conv8_3', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block8_norm3')(o)
    
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='conv8_4', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block8_norm4')(o)

    # upsampling block 9
    o = concatenate([o, f3], axis=channel_axis)
    o = UpSampling2D(name='upsampling9', size=(2,2))(o)
    
    o = Conv2D(256, (3,3), activation='relu', padding='same', name='conv9_1', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block9_norm1')(o)
    
    o = Conv2D(256, (3,3), activation='relu', padding='same', name='conv9_2', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block9_norm2')(o)
    
    o = Conv2D(256, (3,3), activation='relu', padding='same', name='conv9_3', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block9_norm3')(o)
    
    o = Conv2D(256, (3,3), activation='relu', padding='same', name='conv9_4', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block9_norm4')(o)

    # upsampling block 10
    o = concatenate([o, f2], axis=channel_axis)
    o = UpSampling2D(name='upsampling10', size=(2,2))(o)
    
    o = Conv2D(128, (3,3), activation='relu', padding='same', name='conv10_1', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block10_norm1')(o)
    
    o = Conv2D(128, (3,3), activation='relu', padding='same', name='conv10_2', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block10_norm2')(o)

    # upsampling block 11
    o = concatenate([o, f1], axis=channel_axis)
    o = UpSampling2D(name='upsampling11', size=(2,2))(o)
    
    o = Conv2D(64, (3,3), activation='relu', padding='same', name='conv11_1', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block11_norm1')(o)
    
    o = Conv2D(64, (3,3), activation='relu', padding='same', name='conv11_2', data_format=image_ordering)(o)
    o = BatchNormalization(axis=channel_axis, name='block11_norm2')(o)
    
    #Capa de salida
    o = Conv2D(1, (1,1), activation='sigmoid', padding='same', name='output', data_format=image_ordering)(o) #activation='softmax'
    
    model = Model(inputs=input_img, outputs=o)
    
    return model