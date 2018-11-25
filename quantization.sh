
#bazel-bin/tensorflow/tools/quantization/quantize_graph \
#  --input=/home/ubuntu/faststyle/models/optimize_model.pb \
#  --output=/home/ubuntu/faststyle/models/eight_model.pb \
#  --output_node_names="img_t_net/output" \
#  --mode=eightbit


#bazel-bin/tensorflow/tools/quantization/quantize_graph \
#  --input=/home/ubuntu/proteus-deeplab/model/deeplabv3_pascal_trainval/optimize_model.pb \
#  --output=/home/ubuntu/proteus-deeplab/model/deeplabv3_pascal_trainval/eight_model.pb \
#  --output_node_names="SemanticPredictions" \
#  --mode=eightbit

bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=/home/ubuntu/proteus-deeplab/mobilenet_train/optimize_mobilenet_$1.pb \
  --out_graph=/home/ubuntu/proteus-deeplab/mobilenet_train/eight_mobilenet_$1.pb \
  --inputs='ImageTensor:0' --outputs='SemanticPredictions:0' \
  --transforms='
  add_default_attributes
  strip_unused_nodes(type=float, shape="1,299,299,3")
  remove_nodes(op=Identity, op=CheckNumerics)
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms
  quantize_weights
  quantize_nodes
  strip_unused_nodes
  sort_by_execution_order'


