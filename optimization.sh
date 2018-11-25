bazel-bin/tensorflow/python/tools/optimize_for_inference \
  --input=/home/ubuntu/proteus-deeplab/mobilenet_train/frozen_mobilenet_$1.pb \
  --output=/home/ubuntu/proteus-deeplab/mobilenet_train/optimize_mobilenet_$1.pb \
  --input_names="ImageTensor" \
  --output_names="SemanticPredictions"

#bazel-bin/tensorflow/python/tools/optimize_for_inference \
#  --input=/home/ubuntu/proteus-deeplab/train/frozen_model.pb \
#  --output=/home/ubuntu/proteus-deeplab/train/optimize_model.pb \
#  --input_names="ImageTensor" \
#  --output_names="SemanticPredictions"


#bazel-bin/tensorflow/python/tools/optimize_for_inference \
#  --input=/home/ubuntu/faststyle/models/frozen_model.pb \
#  --output=/home/ubuntu/faststyle/models/optimize_model.pb \
#  --input_names="img_t_net/input" \
#  --output_names="img_t_net/output"


