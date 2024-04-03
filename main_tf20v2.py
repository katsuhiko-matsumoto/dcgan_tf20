import tensorflow as tf
import numpy as np
import os
import time
from glob import glob
import logging
import datetime
import sys
import argparse

#orig libs
#from models.model_nondeep256 import DCGAN_NONDEEP256 as DCGAN
from models.model_nondeep64 import DCGAN_NONDEEP64 as DCGAN
#from models.model_deep64v2 import DCGAN_DEEP64v2 as DCGAN
#from models.model_deep256v2 import DCGAN_DEEP256V2 as DCGAN
#from models.model_deep64v2 import DCGAN_DEEP64v2 as DCGAN
from common.utils import *

#------------PARM SETTING--------------
#log-level
log_level = logging.INFO

#log directory
log_dir = 'logs'
#input directory of learning picture
pic_dir = 'data/256_celebA2020'
#checkpoint directory
ckpt_dir = 'training_checkpoints'
#output directory of generated picture
gen_pic_dir = 'out'

#save checkpoint interval(unit:epoch)
ckpt_num = 1
#checkpoint file max max_to_keep
max_to_keep=2
#generate picture interval(unit:epoch)
save_pic_num = 1
#number of sample generated images included in one image
num_examples_to_generate = 64

#epoch
EPOCHS = 20000
#batch size to learn picture
BATCH_SIZE = 64

#training hyper parameters
noise_dim = 100
#Learning rate of for adam [0.0002]
learning_rate=0.0002
#Momentum term of adam [0.5]
beta1=0.5

#for google colab prefix
#add_dir_prefix='dcgan_tf20/'
add_dir_prefix=''

#--------------------------
parser = argparse.ArgumentParser(description='DCGAN')
parser.add_argument('--runmode', required=True, help='rum mode [first, again, generate], first=at the first time learning. again=start from checkpoint. generate=genrate picture from checkpoint', choices=['first','again','generate'])
parser.add_argument('--log_dir', help='log directory')
parser.add_argument('--ckpt_dir', help='checkpoint directory')
parser.add_argument('--train_input_dir', help='input directory of learning picture for training')
parser.add_argument('--output_dir', help='output directory of generated picture')
parser.add_argument('--save_ckpt_num', type=int, help='save checkpoint interval(unit:epoch)')
parser.add_argument('--ckpt_keep_num', type=int, help='checkpoint file max max_to_keep')
parser.add_argument('--save_pic_num', type=int, help='generate picture interval(unit:epoch)')
parser.add_argument('--epochs_num', type=int,help='number of times epoch')
parser.add_argument('--batch_size', type=int, help='batch size to learn picture')

args = parser.parse_args()
if args.log_dir is not None:
    log_dir = args.log_dir
if args.ckpt_dir is not None:
    ckpt_dir = args.ckpt_dir
if args.train_input_dir is not None:
    pic_dir = args.train_input_dir
if args.output_dir is not None:
    gen_pic_dir = args.output_dir
if args.save_ckpt_num is not None:
    ckpt_num = args.save_ckpt_num
if args.ckpt_keep_num is not None:
    max_to_keep = args.ckpt_keep_num
if args.save_pic_num is not None:
    save_pic_num = args.save_pic_num
if args.epochs_num is not None:
    EPOCHS = args.epochs_num
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size

ERR_FLG = False
log_dir = add_dir_prefix+log_dir
if os.path.isdir(os.path.join(log_dir)) == False:
    os.makedirs(os.path.join(log_dir))
log_prefix = os.path.join(log_dir, "system-{}.log".format(timestamp()))
logging.basicConfig(filename=log_prefix, level=log_level)

input_fname_pattern = '*.jpg'
_data_path = os.path.join(add_dir_prefix+pic_dir)
data_path = os.path.join(add_dir_prefix+pic_dir, input_fname_pattern)
if os.path.isdir(_data_path) == False:
    print("ERROR:DIRECTORY is not found : {}".format(_data_path))
    ERR_FLG = True
data = glob(data_path)
if len(data) == 0:
    print("ERROR:[!] No data found in '" + data_path + "'")
    ERR_FLG = True

checkpoint_prefix = os.path.join(add_dir_prefix+ckpt_dir)
if os.path.isdir(checkpoint_prefix) == False:
    os.makedirs(checkpoint_prefix)

gen_pic_dir = os.path.join(add_dir_prefix+gen_pic_dir)
if os.path.isdir(gen_pic_dir) == False:
    os.makedirs(gen_pic_dir)

if ERR_FLG == True:
    print("please fix error. [program exit]")
    #sys.stdout.write(str(1))
    sys.exit()

dcgan = DCGAN()

np.random.shuffle(data)
imreadImg = imread(data[0])
if len(imreadImg.shape) >= 3:
    dcgan.set_cdim(imread(data[0]).shape[-1])
else:
    dcgan.set_cdim(1)

dcgan.gen_gene_and_disc()
generator = dcgan.get_generator()
generator.summary(print_fn=lambda x: logging.info('{}'.format(x)))
generator.summary()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

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

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=max_to_keep)

seed = tf.random.normal([num_examples_to_generate, noise_dim])

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

    return gen_loss, disc_loss, real_output, fake_output

def train(epochs):
    for epoch in range(epochs):
        start = time.time()
        cnt = 0
        np.random.shuffle(data)
        BUFFER_SIZE = len(data)
        if BUFFER_SIZE < BATCH_SIZE:
            logging.error ("[!] Entire dataset size is less than the configured batch_size")
            raise Exception("[!] Entire dataset size is less than the configured batch_size")
        batch_idxs = BUFFER_SIZE // BATCH_SIZE
        for idx in range (int(batch_idxs)):
            batch_start = time.time()
            sample_files = data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            sample = [
                get_image(sample_file,
                      input_height=dcgan.get_input_height(),
                      input_width=dcgan.get_input_width(),
                      resize_height=dcgan.get_output_height(),
                      resize_width=dcgan.get_output_width(),
                      crop=True) for sample_file in sample_files]

            dataset = tf.data.Dataset.from_tensor_slices(sample).shuffle(BATCH_SIZE).batch(BATCH_SIZE)
            for image_batch in dataset:
                gen_loss, disc_loss, real_out, fake_out = train_step(image_batch)
                logging.info('gen_loss:{}'.format(gen_loss.numpy()))
                logging.info('disc_loss:{}'.format(disc_loss.numpy()))
                #logging.info('real_output:{}'.format(real_out))
                #logging.info('fake_output:{}'.format(fake_out))
            #print ('Batch Step(size:{}) :{}/{} in epoch {} is {} sec'.format(BATCH_SIZE, idx+1, batch_idxs, epoch+1, time.time()-batch_start))
            logging.info ('Batch Step(size:{}) :{}/{} in epoch {} is {} sec'.format(BATCH_SIZE, idx+1, batch_idxs, epoch+1, time.time()-batch_start))
        if (epoch + 1) % save_pic_num == 0:
            generate_and_save_images(generator,
                             epoch + 1,
                             seed)
        # Save the model
        if (epoch + 1) % ckpt_num == 0:
            logging.info('save checkpoint:{}'.format(checkpoint_prefix))
            #print('save checkpoint:{}'.format(checkpoint_prefix))
            manager.save()
            #checkpoint.save(file_prefix = checkpoint_prefix)

        #print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        logging.info ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    save_images(predictions, image_manifold_size(predictions.shape[0]),
        '{}/train_{:08d}_{}.png'.format(gen_pic_dir, epoch, timestamp()))
    logging.info("image saved!")
    #print("image saved!")

def load(checkpoint_dir):
    #import re
    logging.info(" [*] Reading checkpoints...{}".format(checkpoint_dir))
    print(" [*] Reading checkpoints...{}".format(checkpoint_dir))
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

def main(args):
    if args.runmode == 'again' or args.runmode == 'first' or args.runmode == 'generate':
        if args.runmode == 'again':
            flag, counter = load(checkpoint_prefix)
            if flag:
                logging.info("# re-learning start")
                print("# re-learning start")
                try:
                    train(EPOCHS)
                except BaseException as e:
                    print(e)
                    logging.error(e, stack_info=True)
            else:
                logging.error("stop. reason:failed to load")
                print("stop. reason:failed to load")
        elif args.runmode == 'first':
            logging.info("# first learning start")
            print("# first learning start")
            try:
                train(EPOCHS)
            except BaseException as e:
                print(e)
                logging.error(e, stack_info=True)
        elif args.runmode == 'generate':
            flag, counter = load(checkpoint_prefix)
            if flag:
                logging.info("# re-learning start")
                print("# image-generate start")
                try:
                    seed = tf.random.normal([num_examples_to_generate, noise_dim])
                    generate_and_save_images(generator,
                                             999,
                                             seed)
                except BaseException as e:
                    print(e)
                    logging.error(e, stack_info=True)

if __name__ == '__main__':
    #args = sys.argv
    #args.runmode='first'
    main(args)
