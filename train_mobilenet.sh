
PATH_TO_INITIAL_CHECKPOINT="model/deeplabv3_pascal_trainval"
PATH_TO_TRAIN_DIR="train"
PATH_TO_DATASET="datasets/pascal_voc_seg"

# From tensorflow/models/research/
python train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="mobilenet_v2" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=4 \
    --tf_initial_checkpoint="model/deeplabv3_mnv2_pascal_trainval/model.ckpt-30000.index" \
    --train_logdir="mobilenet_train" \
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"
