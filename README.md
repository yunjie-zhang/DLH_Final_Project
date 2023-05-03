# DLH_Final_Project
Final Project of DLH

Paper Link: https://arxiv.org/abs/2009.13252
Preprocessed data are located in dataset directory.

How to run the code:
python train_DX.py --predict_type dx --visit_threshold 2 --max_epoch 50 --train_batch_size 32 --valid_visits 10 --num_hidden_layers 1 --embedding_size 120 --dropout 0.1

visit_threshold corresponds to the threshold value in DataPre-processing section.
predict_type can be null. In default, it will perform multi-task training.


