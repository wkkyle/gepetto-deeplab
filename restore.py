import os
import sys
import urllib
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
import argparse

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

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


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--frozen_model_filename", default="model/deeplabv3_pascal_train_aug/frozen_inference_graph.pb", type=str, help="Frozen model file to import")
  args = parser.parse_args()

  #graph = load_graph(args.frozen_model_filename)
  

  for op in graph.get_operations():
    print(op.name)

  INPUT_SIZE = 513

  x = graph.get_tensor_by_name('ImageTensor:0')
  y = graph.get_tensor_by_name('SemanticPredictions:0')

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
	  compat.as_bytes('1'))
	print 'Exporting trained model to', export_path

	builder = saved_model_builder.SavedModelBuilder(export_path)

	tensor_info_x = utils.build_tensor_info(x)
	tensor_info_y = utils.build_tensor_info(y)

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

