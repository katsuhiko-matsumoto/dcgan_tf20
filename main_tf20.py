import tensorflow as tf
tf.__version__

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from glob import glob
import cv2
import scipy.misc
from skimage.transform import resize
from PIL import Image
import imageio
import logging
import datetime

#orig import
#from models.model_nondeep256 import DCGAN_NONDEEP256 as DCGAN
#from models.model_nondeep64 import DCGAN_NONDEEP64 as DCGAN
#from models.model_nondeep64v2 import DCGAN_NONDEEP64V2 as DCGAN
#from models.model_deep256v2 import DCGAN_DEEP256V2 as DCGAN
from models.model_deep64v2 import DCGAN_DEEP64v2 as DCGAN
from common.utils import *

#from IPython import display
#データセットの名前：mnist以外は、オリジナルデータ
dataset_name = 'mnist0'
#学習で使うデータ枚数
sample_num = 128 # 1024
BUFFER_SIZE = sample_num
#バッチサイズ：データ数より大きいとNG
BATCH_SIZE = 128
#モデルのセーブデータ：チェックポイント間隔：単位EPOCH
ckpt_num = 5
#画像生成間隔：単位EPOCH
save_pic_num = 1
#生成される一枚の画像の含まれる顔画像の数
num_examples_to_generate = 64
#実行モード: again(チェックポイントから再開) or first(初実行)
exec_mode = 'again'

#for google colab prefix
#add_dir_prefix='dcgan_tf20/'
add_dir_prefix=''

#training hyper parameters
EPOCHS = 20000
noise_dim = 100
#Learning rate of for adam [0.0002]
learning_rate=0.0002
#Momentum term of adam [0.5]
beta1=0.5

dcgan = DCGAN()

log_dir = add_dir_prefix+'logs'
log_prefix = os.path.join(log_dir, "sysetm-{}.log".format(timestamp()))
logging.basicConfig(filename=log_prefix, level=logging.INFO)

if dataset_name == 'mnist':
    #dataset
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    logging.info('##train_images:{}'.format(train_images[0]))
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float64')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    logging.info('##dataset:{}'.format(train_dataset))
    logging.info('#dataset len:{}'.format(len(train_dataset)))
else:
    data_dir = add_dir_prefix+'data'
    dataset_name = '256_celebA2020'
    input_fname_pattern = '*.jpg'

    data_path = os.path.join(data_dir, dataset_name, input_fname_pattern)
    data = glob(data_path)
    if len(data) == 0:
      raise Exception("[!] No data found in '" + data_path + "'")
    np.random.shuffle(data)
    imreadImg = imread(data[0])
    if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
      dcgan.set_cdim(imread(data[0]).shape[-1])
    else:
      dcgan.set_cdim(1)

    if len(data) < BATCH_SIZE:
      raise Exception("[!] Entire dataset size is less than the configured batch_size")
    sample_files = data[0:sample_num]
    sample = [
        get_image(sample_file,
                  input_height=dcgan.get_input_height(),
                  input_width=dcgan.get_input_width(),
                  resize_height=dcgan.get_output_height(),
                  resize_width=dcgan.get_output_width(),
                  crop=True,
                  grayscale=False) for sample_file in sample_files]

    train_dataset = tf.data.Dataset.from_tensor_slices(sample).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    logging.info('##dataset:{}'.format(train_dataset))

dcgan.gen_gene_and_disc()
generator = dcgan.get_generator()
generator.summary(print_fn=lambda x: logging.info('{}'.format(x)))
generator.summary()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

#損失
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

discriminator = dcgan.get_discriminator()
discriminator.summary(print_fn=lambda x: logging.info('{}'.format(x)))
discriminator.summary()
decision = discriminator(generated_image)
logging.info ("##decision:{}".format(decision))

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta1)
#generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
#discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
# 1e-4 = 0.0004
#generator_optimizer = tf.keras.optimizers.Adam(1e-4)
#discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#save checkpoint
checkpoint_dir = add_dir_prefix+'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    cnt = 0
    for image_batch in dataset:
      cnt = cnt+1
      logging.info ('Learning Step :{}'.format(cnt))
      gen_loss, disc_loss = train_step(image_batch)
      logging.info('gen_loss:{}'.format(gen_loss.numpy()))
      logging.info('disc_loss:{}'.format(disc_loss.numpy()))
    # Produce images for the GIF as you go
    #display.clear_output(wait=True)
    if (epoch + 1) % save_pic_num == 0:
      generate_and_save_images(generator,
                             epoch + 1,
                             seed)
    # Save the model every 5 epochs
    if (epoch + 1) % ckpt_num == 0:
      logging.info('save checkpoint:{}'.format(checkpoint_prefix))
      print('save checkpoint:{}'.format(checkpoint_prefix))
      checkpoint.save(file_prefix = checkpoint_prefix)

    logging.info ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  #display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


#画像保存
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  data_path = os.path.join(add_dir_prefix+'out')
  save_images(predictions, image_manifold_size(predictions.shape[0]),
        '{}/train_{:08d}_{}.png'.format(data_path, epoch, timestamp()))
  logging.info("image saved!")
  print("image saved!")

def load(checkpoint_dir):
    #import re
    logging.info(" [*] Reading checkpoints...{}".format(checkpoint_dir))
    print(" [*] Reading checkpoints...{}".format(checkpoint_dir))
    # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    # print("     ->", checkpoint_dir)
    checkpoint_prefix_load = os.path.join(checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_prefix_load)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

      checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix_load))
      #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      counter = int(ckpt_name.split('-')[-1])
      logging.info("******** [*] Success to read {}".format(ckpt_name))
      print("******** [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      logging.error(" [*] Failed to find a checkpoint")
      print(" [*] Failed to find a checkpoint")
      return False, 0

def main(mode):
    if mode == 'again':
        #from checkpoint
        flag, counter = load(checkpoint_dir)
        if flag:
            #exec
            logging.info("# re-learning start")
            print("# re-learning start")
            train(train_dataset, EPOCHS)
        else:
            logging.error("stop. reason:failed to load")
            print("stop. reason:failed to load")
    elif mode == 'first':
        logging.info("# first learning start")
        print("# first learning start")
        train(train_dataset, EPOCHS)

if __name__ == '__main__':
    main(exec_mode)
