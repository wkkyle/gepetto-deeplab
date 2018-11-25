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

#_MODEL_URLS = {
#    'xception_coco_voctrainaug': 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
#    'xception_coco_voctrainval': 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz',
#}

#Config = collections.namedtuple('Config', 'model_url, model_dir')

#def get_config(model_name, model_dir):
#    return Config(_MODEL_URLS[model_name], model_dir)

#config_widget = interactive(get_config, model_name=_MODEL_URLS.keys(), model_dir='')
#display.display(config_widget)

_TARBALL_NAME = 'deeplab_model.tar.gz'

#config = config_widget.result

#model_dir = config.model_dir or tempfile.mkdtemp()
#tf.gfile.MakeDirs(model_dir)

#download_path = os.path.join(model_dir, _TARBALL_NAME)
#print 'downloading model to %s, this might take a while...' % download_path
#urllib.urlretrieve(config.model_url, download_path)
#print 'download completed!'
download_path = os.path.join('model', _TARBALL_NAME)

MODEL_NAME = 'model/deeplabv3_pascal_train_aug'
_FROZEN_GRAPH_NAME = MODEL_NAME + '/frozen_inference_graph.pb'
#_FROZEN_GRAPH_NAME = 'new_train/frozen_model.pb'
_FROZEN_GRAPH_NAME = 'model/deeplabv3_pascal_train_aug/frozen_model.pb'
#_FROZEN_GRAPH_NAME = 'model/deeplabv3_pascal_train_aug/frozen_inference_graph.pb'
#_FROZEN_GRAPH_NAME = 'frozen_inference_graph'


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
import cPickle as pickle
import gzip

def main():

    model = DeepLabModel(download_path)

    img, seg_map = run_demo_image(model, 'park.jpg')

    #print(seg_map)
    pickle.dump(seg_map, gzip.open('suzi.pkl','wb'))
    segMap = pickle.load(gzip.open('suzi.pkl','rb'))

    seg_image = get_dataset_colormap.label_to_color_image(
        segMap, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
        #seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    cv2.imwrite('img/seg1.jpg',seg_image)

    #print(type(img), type(seg_image))
    img = np.clip(img,0,255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    overlay = cv2.addWeighted(img, 0.3, seg_image, 0.7,0)
    cv2.imwrite('img/overlay1.jpg',overlay)

    blur = cv2.GaussianBlur(img, (9,9), 0)

    seg2gray = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(seg2gray, 5, 255, cv2.THRESH_BINARY)
    cv2.imwrite('img/mask.jpg',mask)
    mask_inv = cv2.bitwise_not(mask)

    fg = cv2.bitwise_and(img, img, mask=mask)
    bg = cv2.bitwise_and(blur, blur, mask=mask_inv)
    graybg = cv2.bitwise_and(gray, gray, mask=mask_inv)

    dst = cv2.add(fg, bg) 
    cv2.imwrite('img/blur.jpg',dst)

    graybg = cv2.cvtColor(graybg, cv2.COLOR_GRAY2BGR)
    dst = cv2.add(fg, graybg) 
    cv2.imwrite('img/gray.jpg',dst)



test='''
    img, seg_map = run_demo_image(model, 'image2.jpg')
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    #cv2.imwrite('img/seg2.jpg',seg_image)
    img = np.clip(img,0,255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img, 0.3, seg_image, 0.7,0)
    cv2.imwrite('img/overlay2.jpg',overlay)

    img, seg_map = run_demo_image(model, 'image3.jpg')
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    #cv2.imwrite('img/seg3.jpg',seg_image)
    img = np.clip(img,0,255).astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img, 0.3, seg_image, 0.7,0)
    cv2.imwrite('img/overlay3.jpg',overlay)
'''

if __name__ == '__main__':
    main()
