

CKPT_PATH="model/deeplabv3_pascal_train_aug/model.ckpt"
EXPORT_PATH="model/deeplabv3_pascal_train_aug/frozen_model.pb"


python export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=21 \
  --crop_size=960 \
  --crop_size=960 \
  --inference_scales=1.0

exit 1


python export_model.py \
  --checkpoint_path="model/deeplabv3_pascal_train_aug/model.ckpt" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --crop_size=513 \
  --crop_size=513 \
  --export_path="model/deeplabv3_pascal_train_aug/frozen_model.pb"

#model/deeplabv3_mnv2_pascal_trainval
  #--checkpoint_path="train/model.ckpt-30000" \
  #--checkpoint_path="new_train/model.ckpt-3749" \
  #--export_path="new_train/frozen_model.pb"
  #--checkpoint_path="model/deeplabv3_mnv2_pascal_trainval/model.ckpt-30000" \
  #--export_path="model/deeplabv3_mnv2_pascal_trainval/frozen_model.pb"
