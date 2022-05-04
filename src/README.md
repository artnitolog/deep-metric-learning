# Запуск обучения и инференса

* Для запуска необходимо воспользоваться скриптом `run.py`.

Пример запуска:

```python
python run.py \
--embedding_dim 256 \
--model_name bert \
--dataset_name news20 \
--loss_name ContrastiveLoss \
--batch_size 32 \
--inference_batch_size 128 \
--num_epochs 10 \
--iters_per_epoch 0 \
--patience_epochs 10 \
--lr 1e-4 \
--warmup_iters 100 \
--num_workers 0 \
--samples_per_class 1
```