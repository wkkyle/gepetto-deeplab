python export_model.py \
  --checkpoint_path="mobilenet_train/model.ckpt-30000" \
  --model_variant="mobilenet_v2" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --crop_size=200 \
  --crop_size=200 \
  --export_path="mobilenet_train/frozen_mobilenet_200.pb"


#model/deeplabv3_mnv2_pascal_trainval
  #--checkpoint_path="train/model.ckpt-30000" \
  #--checkpoint_path="model/deeplabv3_mnv2_pascal_trainval/model.ckpt-30000" \
  #--export_path="model/deeplabv3_mnv2_pascal_trainval/frozen_model.pb"
