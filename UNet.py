import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

def inference(img1,img2):
    assert img is not None
    with tf.compat.v1.variable_scope("inference"):
        # Conv1 + MaxPooling1 + Crop1
        # BF of SAR images
        conv1_1_BF = tf.layers.conv2d(img1, 64, 3, activation=tf.nn.relu, padding='same',
           kernel_initializer = tf.contrib.layers.xavier_initializer())  

        conv1_2_BF = tf.layers.conv2d(conv1_1_BF, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())

        #PF of SAR images
        cov1_1_PF = tf.layers.conv2d(img2, 64, 3, activation=tf.nn.relu, padding='same',
           kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv1_2_PF = tf.layers.conv2d(conv1_1_PF, 64, 3, activation=tf.nn.relu, padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        # merge BF and PF
        merge1 = tf.concat([conv1_2_BF, conv1_2_PF], axis=3)

        # pooling
        pool1 = tf.layers.max_pooling2d(merge1, pool_size=2, strides=2, padding='same')
 

        conv1_merge1 = tf.layers.conv2d(pool1, 256, 3, activation=tf.nn.relu, padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        # img为输入图像，为一个tensor，具有[batch,in_height,in_width,in_channels]，即图片数量，图片高度，图片宽度，图像通道数
        # 64为filters数量，即卷积中的滤波器数
        # 3为滤波器大小，即为3*3卷积核
        # activation为激活功能，即使用分段线性函数进行激活
        # padding=same表示图像进行填充，使得滤波器可以达到图像边缘
        # kernel_initializer为卷积内核的初始化程序，返回一个初始化权重矩阵

        conv1_merge2 = tf.layers.conv2d(conv1_merge1, 256, 3, activation=tf.nn.relu, padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv1_merge2, pool_size=2, strides=2, padding='same')
        crop1 = tf.keras.layers.Cropping2D(cropping=((88, 88), (88, 88)))(conv1_2_2)
        #corpping2D函数对图像进行裁剪，(top_crop=88,bottom_crop=88),(left_crop=88,right_crop=88)
        
        # Conv2 + MaxPooling2 + Crop2
        conv2_1 = tf.layers.conv2d(pool2, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv2_2 = tf.layers.conv2d(conv2_1, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=2, strides=2, padding='same')
        crop2 = tf.keras.layers.Cropping2D(cropping=((40, 40), (40, 40)))(conv2_2)

        
        # Conv1-Conv5为Encoder编码层

        # UpConv + Concat + Conv6
        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv2_2)
        #UpSampling2D对图片数据在高和宽的方向进行数据插值
        
        # merge6 = tf.concat([crop4, up6], axis=3) # concat channel
        merge3 = tf.concat([conv2_2, up6], axis=3)  # concat channel
        #将conv4_2与up6层融合，即unet中的特征融合模块，axis=3指沿第3维进行融合
        
        conv6_1 = tf.layers.conv2d(merge3, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv6_2 = tf.layers.conv2d(conv6_1, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # UpConv + Concat + Conv7
        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6_2)
        # merge7 = tf.concat([crop3, up7], axis=3) # concat channel
        merge7 = tf.concat([conv3_2, up7], axis=3)  # concat channel
        conv7_1 = tf.layers.conv2d(merge7, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv7_2 = tf.layers.conv2d(conv7_1, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())

        # UpConv + Concat + Conv8
        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_2)
        # merge8 = tf.concat([crop2, up8], axis=3) # concat channel
        merge8 = tf.concat([conv2_2, up8], axis=3)  # concat channel
        conv8_1 = tf.layers.conv2d(merge8, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv8_2 = tf.layers.conv2d(conv8_1, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())

        # UpConv + Concat + Conv9

        final = tf.layers.conv2d(conv8_2, 1, 1, activation=tf.nn.sigmoid, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())

    return final