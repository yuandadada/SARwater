import tensorflow as tf
import numpy as np
import sys
import os
import glob
import UNet
from PIL import Image
import random
import datetime
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt


class AttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size=1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

    def forward(self, x, g):
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.psi(nn.ReLU()(x1 + g1))
        return out * x


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(out_channels, in_channels, int(out_channels / 2))
        self.conv_bn1 = ConvBatchNorm(in_channels + out_channels, out_channels)
        self.conv_bn2 = ConvBatchNorm(out_channels, out_channels)

    def forward(self, x, x_skip):
        # note : x_skip is the skip connection and x is the input from the previous block
        # apply the attention block to the skip connection, using x as context
        x_attention = self.attention(x_skip, x)
        # upsample x to have th same size as the attention map
        x = nn.functional.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=False)
        # stack their channels to feed to both convolution blocks
        x = torch.cat((x_attention, x), dim=1)
        x = self.conv_bn1(x)
        return self.conv_bn2(x)


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)  # ??????N=4???????????????
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)  # ??????1x1??????????????????channel????????????1/N
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)  # ?????????1x1????????????????????????channel

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)  # ?????????????????????????????????size?????????1x1,2x2,3x3,6x6
        return avgpool(x)

    def upsample(self, x, size):  # ??????????????????????????????
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)  # concat ?????????????????????
        x = self.out(x)
        return x

def train(loss, var):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.lr) #tf.train.AdamOptimizer???Adam?????????????????????????????????????????????????????????????????????
    grads = optimizer.compute_gradients(loss, var_list=var) #optimizer.compute_gradients??????loss???????????????????????????
    return optimizer.apply_gradients(grads) #optimizer.apply_gradients()???????????????????????????????????????????????????

def load(img_list, mask_list, i):
    origin_img = Image.open(img_list[i])
    origin_mask = Image.open(mask_list[i])
    input_img = np.expand_dims(np.expand_dims(np.array(origin_img), axis=0), axis=3) #np.expand_dims???????????????????????????axis??????????????????????????????origin_img?????????????????????????????????tf.layers.conv2d????????????
    input_mask = np.uint16(np.expand_dims(np.expand_dims(np.array(origin_mask), axis=0), axis=3) / 255)
    return input_img, input_mask, origin_img, origin_mask

def main(_):
    FLAGS = tf.flags.FLAGS
    # gpu config.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    #??????????????????

    # tf.compat.v1.disable_eager_execution()
    image = tf.compat.v1.placeholder(tf.float32, shape=[None, 1024, 1024, 1], name="image") #????????????,tf.float32????????????????????????shpe?????????????????????name?????????
    mask = tf.compat.v1.placeholder(tf.int32, shape=[None, 1024, 1024, 1], name="mask")

    train_model = UNet.inference(image) #???????????????????????????
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(mask, train_model)) #tf.reduce_mean?????????????????????????????????????????????????????????????????????tensor???????????????????????????????????????????????????;
                                                                                  # tf.keras.losses.binary_crossentropy?????????????????????????????????????????????????????????

    trainable_var = tf.compat.v1.trainable_variables() #????????????????????????
    train_op = train(loss, trainable_var)

    saver = tf.compat.v1.train.Saver(max_to_keep=FLAGS.epochs) #tf.compat.v1.train.Saver???????????????

    if FLAGS.mode == "train":
        with tf.compat.v1.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.weight_dir, str(FLAGS.threshold)))
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path) #????????????
            else:
                sess.run(tf.compat.v1.global_variables_initializer()) #??????????????????????????????
            #?????????????????????????????????;????????????????????????????????????

            img_list = sorted(glob.glob(FLAGS.training_set + '/img/*.tif')) #glob.glob()????????????????????????????????????????????????list????????????;sorted????????????????????????????????????
            mask_list = sorted(glob.glob(FLAGS.training_set + '/mask/*.tif'))

            val_img_list = sorted(glob.glob(FLAGS.validation_set + '/img/*.tif'))
            val_mask_list = sorted(glob.glob(FLAGS.validation_set + '/mask/*.tif'))

            train_data_size = len(img_list)
            train_iterations = int(train_data_size/FLAGS.bs)
            val_data_size = len(val_img_list)

            for cur_epoch in range(FLAGS.epochs):
                # Shuffle
                tmp = [[x,y] for x,y in zip(img_list, mask_list)]
                random.shuffle(tmp)
                img_list = [n[0] for n in tmp]
                mask_list = [n[1] for n in tmp]

                now = datetime.datetime.now()
                print(now)
                print("threshold: ", FLAGS.threshold, " Start training!, epoch: ", cur_epoch)
                for i in range(train_data_size):
                    # print('i : ',i)
                    input_img, input_mask,_,_ = load(img_list, mask_list, i)

                    feed_dict = {image:input_img, mask:input_mask}
                    sess.run(train_op, feed_dict=feed_dict)

                # validation
                if cur_epoch % 1 ==0:
                    avg_loss = 0
                    iou_score_list = []
                    for i in range(val_data_size):
                        input_img, input_mask,_,_ = load(val_img_list, val_mask_list, i)

                        feed_dict = {image: input_img, mask: input_mask}
                        loss1, pred = sess.run([loss, train_model], feed_dict=feed_dict)
                        avg_loss += loss1

                        # save result
                        pred1 = np.where(pred < FLAGS.threshold, 0.0, 1.0)
                        ttt = Image.fromarray(pred1[0, :, :, 0])
                        img_save_path = os.path.join(FLAGS.result, str(FLAGS.threshold))
                        if not os.path.exists(img_save_path):
                            os.makedirs(img_save_path)
                        file_name = img_save_path +'/epoch' + '%02d' % cur_epoch + '_' + val_mask_list[i].split('/')[-1]
                        ttt.save(file_name)

                        # calculate IoU
                        if np.sum(input_mask) == 0:
                            input_mask = np.logical_not(input_mask)
                            pred1 = np.logical_not(pred1)
                        intersection = np.logical_and(input_mask, pred1)
                        union = np.logical_or(input_mask, pred1)
                        iou_score_list.append(np.sum(intersection) / np.sum(union))

                avg_loss = avg_loss/val_data_size
                print('epoch: ', cur_epoch, 'average loss: ',avg_loss)
                print(iou_score_list)
                print('average iou: ', sum(iou_score_list) / len(iou_score_list))

                weight_save_path = os.path.join(FLAGS.weight_dir, str(FLAGS.threshold), FLAGS.weight)
                if not os.path.exists(weight_save_path):
                    os.makedirs(weight_save_path)
                saver.save(sess, weight_save_path, global_step=cur_epoch)

    if FLAGS.mode == "val":
        val_img_list = sorted(glob.glob(FLAGS.validation_set + '/img/*.tif'))
        val_mask_list = sorted(glob.glob(FLAGS.validation_set + '/mask/*.tif'))

        val_data_size = len(val_img_list)

        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.weight_dir, str(FLAGS.threshold)))
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sys.exit("No weights!!")
            avg_loss = 0
            iou_score_list = []
            for i in range(val_data_size):
                input_img, input_mask, origin_img, origin_mask = load(val_img_list, val_mask_list, i)
                feed_dict = {image: input_img, mask: input_mask}
                loss1, pred = sess.run([loss, train_model], feed_dict=feed_dict)
                pred1 = np.where(pred < FLAGS.threshold, 0, 1)
                avg_loss += loss1
                # display
                # image_show = Image.fromarray(np.uint8(origin_img)).convert('RGB')
                # image_show = image_show.resize((400,400))
                # image_show.show()
                fig = plt.figure(figsize=(16, 8))
                rows = 1
                cols = 2

                ax1 = fig.add_subplot(rows, cols, 1)
                ax1.imshow(input_mask[0, :, :, 0], cmap='gray')
                ax1.set_title('Ground truth')
                ax1.axis("off")

                ax2 = fig.add_subplot(rows, cols, 2)
                ax2.imshow(pred1[0, :, :, 0], cmap='gray')
                ax2.set_title('Predicted result')
                ax2.axis("off")

                plt.show(block=False)
                plt.pause(3)  # 3 seconds
                plt.close()

                # calculate IoU
                if np.sum(input_mask)==0:
                    input_mask = np.logical_not(input_mask)
                    pred1 = np.logical_not(pred1)
                intersection = np.logical_and(input_mask, pred1)
                union = np.logical_or(input_mask, pred1)
                iou_score_list.append(np.sum(intersection) / np.sum(union))

            print(iou_score_list)
            print('average iou: ', sum(iou_score_list)/len(iou_score_list))


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string("mode", "val", "mode: train/val/test")
    tf.flags.DEFINE_string("weight_dir", "./weight_UNet", "weight_FCN_VGG16 directory.")
    tf.flags.DEFINE_string("weight_FCN_VGG16", "UNet", "the latest weight_FCN_VGG16 saved.")
    tf.flags.DEFINE_float("lr", "1e-4", "learning rate.")
    tf.flags.DEFINE_float("threshold", "0.5", "threshold.")
    tf.flags.DEFINE_string("training_set", "./data/train_ver2", "dataset path for training.")
    tf.flags.DEFINE_string("validation_set", "./data/val", "dataset path for validation.")
    tf.flags.DEFINE_string("result", "./data/val/result_UNet", "path for validation result.")
    tf.flags.DEFINE_integer("bs", 32, "batch size for training.")
    tf.flags.DEFINE_integer("epochs", 100, "total training epochs.")

    tf.compat.v1.app.run(main=main)