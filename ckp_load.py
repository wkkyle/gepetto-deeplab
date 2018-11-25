import collections
import os
import StringIO
import sys
import tarfile
import tempfile
import urllib

from IPython import display
from ipywidgets import interact
from ipywidgets import interactive
from matplotlib import gridspec
#from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

sys.path.append('utils')
import get_dataset_colormap

MODEL_NAME = 'model/deeplabv3_pascal_train_aug'
_FROZEN_GRAPH_NAME = MODEL_NAME + '/frozen_inference_graph.pb'


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        
        graph_def = None

        # Extract frozen graph from tar archive.
        #tar_file = tarfile.open(tarball_path)
        #for tar_info in tar_file.getmembers():
        #    if _FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        #        file_handle = tar_file.extractfile(tar_info)
        #        graph_def = tf.GraphDef.FromString(file_handle.read())
        #        break
        #tar_file.close()
        fid = tf.gfile.GFile(_FROZEN_GRAPH_NAME)
        graph_def = tf.GraphDef.FromString(fid.read())
        
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():      
            tf.import_graph_def(graph_def, name='')
        
        self.sess = tf.Session(graph=self.graph)
            
    def run(self, image):
        """Runs inference on a single image.
        
        Args:
            image: A PIL.Image object, raw input image.
            
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)

def vis_segmentation(image, seg_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')
    
    plt.subplot(grid_spec[1])
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')
    
    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0)

#IMAGE_DIR = 'g3doc/img'
IMAGE_DIR = ''

def run_demo_image(model, image_name):
    try:
        image_path = os.path.join(IMAGE_DIR, image_name)
        orignal_im = Image.open(image_path)
    except IOError:
        print 'Failed to read image from %s.' % image_path 
        return 
    print 'running deeplab on image %s...' % image_name
    resized_im, seg_map = model.run(orignal_im)
    
    #vis_segmentation(resized_im, seg_map)
    return resized_im, seg_map

import cv2

def main():

    model = DeepLabModel(download_path)

    img, seg_map = run_demo_image(model, 'mb.jpg')
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    #cv2.imwrite('img/seg1.jpg',seg_image)
    #print(type(img), type(seg_image))
    img = np.clip(img,0,255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img, 0.3, seg_image, 0.7,0)
    cv2.imwrite('img/overlay1.jpg',overlay)

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

import argparse

if __name__ == '__main__':


  parser = argparse.ArgumentParser()
  #parser.add_argument("--frozen_model_filename", default="model/deeplabv3_pascal_train_aug/frozen_inference_graph.pb", type=str, help="Frozen model file to import")
  parser.add_argument("--frozen_model_filename", default="new_train/frozen_model.pb", type=str, help="Frozen model file to import")
  args = parser.parse_args()


  # We use our "load_graph" function
  graph = load_graph(args.frozen_model_filename)

  # We can verify that we can access the list of operations in the graph
  #for op in graph.get_operations():
  #  print(op.name)

  #INPUT_TENSOR_NAME = 'ImageTensor:0'
  #OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  #INPUT_SIZE = 513
  INPUT_SIZE = 960

  #x = graph.get_tensor_by_name('prefix/ImageTensor:0')
  #y = graph.get_tensor_by_name('prefix/SemanticPredictions:0')
  x = graph.get_tensor_by_name('ImageTensor:0')
  y = graph.get_tensor_by_name('SemanticPredictions:0')
  print(x.shape)
  print(y.shape)
  #sys.exit(0)

  image_path = 'mb.jpg'
  img = Image.open(image_path)

  with tf.Session(graph=graph) as sess:
	width, height = img.size
	resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
	target_size = (int(resize_ratio * width), int(resize_ratio * height))
	resized_image = img.convert('RGB').resize(target_size, Image.ANTIALIAS)

	batch_seg_map = sess.run(
	    y,
	    feed_dict={x: [np.asarray(resized_image)]})

	seg_map = batch_seg_map[0]

	export_path_base = 'saved/'
	export_path = os.path.join(
	  compat.as_bytes(export_path_base),
	  compat.as_bytes('3'))
	print 'Exporting trained model to', export_path

	builder = saved_model_builder.SavedModelBuilder(export_path)

	tensor_info_x = utils.build_tensor_info(x)
	#print(tf.shape(tensor_info_x))
	tensor_info_y = utils.build_tensor_info(y)
	#print(tf.shape(tensor_info_y)) 

	prediction_signature = signature_def_utils.build_signature_def(
		inputs={'img': tensor_info_x},
		outputs={'seg': tensor_info_y},
		method_name=signature_constants.PREDICT_METHOD_NAME)

	legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
	builder.add_meta_graph_and_variables(
		sess, [tag_constants.SERVING],
		signature_def_map={
		  'predict_seg': prediction_signature,
		},
		legacy_init_op=legacy_init_op)

	builder.save()




"""
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	#new_saver = tf.train.import_meta_graph('model/deeplabv3_pascal_trainval/model.ckpt.meta')
	#new_saver.restore(sess, 'model/deeplabv3_pascal_trainval/model.ckpt')
	new_saver = tf.train.import_meta_graph('train/model.ckpt-30000.meta')
	new_saver.restore(sess, 'train/model.ckpt-30000')
	g = tf.get_default_graph()
  
	for op in g.get_operations():
		print(op.name)

	#x = g.get_tensor_by_name('ImageTensor:0')
	#y = g.get_tensor_by_name('SemanticPredictions:0')
	#print(x.shape)
	#print(y.shape)

  sys.exit(0)
"""
