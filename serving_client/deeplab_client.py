# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow.core.framework import types_pb2

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

#from tensorflow_serving.example import mnist_input_data

#import Image
import numpy as np
import PIL
from PIL import Image
import cv2
import sys
import os

#sys.path.append('utils')
from utils import get_dataset_colormap

#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from keras.applications.imagenet_utils import preprocess_input

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('model', 'deeplab', 'model name')
tf.app.flags.DEFINE_string('input_dir', '', 'input directory')
tf.app.flags.DEFINE_string('output_dir', 'seg', 'output directory')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


def read_image(path):
  im = Image.open(path)
  imData = list(im.getdata())
  return imData

def readImg(path):
  im = Image.open(path)
  im.load()
  img_in = im.resize((224,224),PIL.Image.ANTIALIAS)
  data = np.asarray( img_in, dtype="float32" )
  return data

def cvimread(path):
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  img_in = cv2.resize(img, (224,224))
  h, w, c = img_in.shape
  img_4d = img_in[np.newaxis, :]
  data = np.asarray( img_4d, dtype="uint8" )
  return data, w, h

import time
#INPUT_SIZE = 513
#INPUT_SIZE = 480
#INPUT_SIZE = 480
INPUT_SIZE = 960
#INPUT_SIZE = 1080

import cPickle as pickle
import gzip
import glob

def load_input(path):
  im = Image.open(path)
  width, height = im.size
  resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
  target_size = (int(resize_ratio * width), int(resize_ratio * height))
  #target_size = (513, 380)
  resized_image = im.convert('RGB').resize(target_size, Image.ANTIALIAS)  
  data = np.asarray( resized_image, dtype="uint8" )
  img_4d = data[np.newaxis, :]

  return img_4d, target_size, resized_image

def main(_):

  output_dir = os.path.expanduser(FLAGS.output_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  start = time.time()
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  start = time.time()
  #data, w, h = cvimread(FLAGS.image)
  #print('img load: ',time.time()-start)
  image_paths = []

  if FLAGS.input_dir != '':
    files = FLAGS.input_dir + '/*.jpg'
    for image_path in glob.glob(files):
      image_paths.append(image_path)
  else:
      image_paths.append(FLAGS.image)

  img_length = len(image_paths)
  batch_shape = (img_length,)+(380, 513, 3)
  #batch_shape = (img_length,)+(INPUT_SIZE, INPUT_SIZE, 3)
  #X_batch = np.zeros(batch_shape, dtype=np.float32) 
  X_batch = np.zeros(batch_shape, dtype=np.uint8) 
  print(X_batch.shape)

  i=0
  for image_path in image_paths:

    #print(w,h)
    #print(data.shape)
    print(image_path)

    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(output_dir, filename+'.pkl')

    #if os.path.isfile(output_filename):
    #  continue

    data, (w, h), resized = load_input(image_path)
    print(w,h)
    #X_batch[i]=data
    i = i + 1

  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model
  request.model_spec.signature_name = 'predict_seg'
  request.inputs['img'].CopyFrom(
    tf.contrib.util.make_tensor_proto(data, shape=(1,h,w,3)))
    #tf.contrib.util.make_tensor_proto(X_batch, shape=X_batch.shape))

  start = time.time()
  out = stub.Predict(request, 30.0)  # 5 secs timeout
  print('network: ',time.time()-start)

  print(out.outputs['seg'].tensor_shape)
  #import sys
  #sys.exit(1)

  shape=[]
  shape.append(out.outputs['seg'].tensor_shape.dim[1].size)
  shape.append(out.outputs['seg'].tensor_shape.dim[2].size)

  seg = np.asarray(out.outputs['seg'].int64_val, np.int64)
  seg = seg.reshape(shape)

  pickle.dump(seg, gzip.open(output_filename,'wb'))
  #segMap = pickle.load(gzip.open(output_filename,'rb'))

  seg_image = get_dataset_colormap.label_to_color_image(
	seg, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
  cv2.imwrite('deeplab/seg_map.jpg',seg_image)

  img = np.clip(resized, 0, 255).astype('uint8')
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  overlay = cv2.addWeighted(img, 0.3, seg_image, 0.7, 0)
  cv2.imwrite('deeplab/overlay.jpg',overlay)

if __name__ == '__main__':
  tf.app.run()
