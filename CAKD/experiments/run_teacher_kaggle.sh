# GD1: fine-tune teacher ViT-B/16 -> 3 lop. Chay tu thu muc CAKD/.
torchrun --nproc_per_node=1 dist_train_teacher.py \
  --data-path /kaggle/working/data_if \
  --batch-size 32 --epochs 15 --lr 2e-4 --amp \
  --output-dir /kaggle/working
