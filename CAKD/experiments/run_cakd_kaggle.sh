# GD2: CAKD distill ViT(3 lop) -> ResNet-50(3 lop). Chay tu thu muc CAKD/.
torchrun --nproc_per_node=1 dist_train_cakd.py \
  --data-path /kaggle/working/data_if \
  --teacher-weights /kaggle/working/teacher_3cls.pth \
  --student-pretrained \
  --batch-size 32 --lr 0.01 --epochs 60 \
  --lr-warmup-epochs 5 --lr-warmup-method linear \
  --distill-start 5 --distill-ramp 20 \
  --auto-augment ta_wide --random-erase 0.1 --mixup-alpha 0.2 \
  --label-smoothing 0.1 --model-ema --amp \
  --train-crop-size 224 --val-resize-size 224 \
  --output-dir /kaggle/working/results
