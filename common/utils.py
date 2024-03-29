#The MIT License (MIT)
#
#Copyright (c) 2016 Taehoon Kim
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import time
import cv2
from skimage.transform import resize
import imageio
import datetime

#---util---------

def imread(path, grayscale = False):
    # Reference: https://github.com/carpedm20/DCGAN-tensorflow/issues/162#issuecomment-315519747
    img_bgr = cv2.imread(path)
    # Reference: https://stackoverflow.com/a/15074748/
    img_rgb = img_bgr[..., ::-1]
    return img_rgb.astype(np.float64)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True):
  image = imread(image_path)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    cropped_image = resize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return resize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])
  #return numpy.array(Image.fromarray(arr).resize(
  #     [resize_h, resize_w], x[j:j+crop_h, i:i+crop_w]))

def gen_random(mode, size):
    if mode=='normal01': return np.random.normal(0,1,size=size)
    if mode=='uniform_signed': return np.random.uniform(-1,1,size=size)
    if mode=='uniform_unsigned': return np.random.uniform(0,1,size=size)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return imageio.imwrite(path, image)

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w

def inverse_transform(images):
  return (images+1.)/2.

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def timestamp(s='%Y%m%d.%H%M%S', ts=None):
  if not ts: ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime(s)
  return st

#----------------------------
