### FT
python ./prior_main.py \
    --method finetune \
    --n_memories_per_class -1 \
    --group final_cifar_experi
    --group final_cifar_experiments \
    --save_each_task_model \
    --epochs_per_task 140 \
    --total_n_memories -1 \
    --use_best_model

### ER-infinite
python ./prior_main.py \
    --method finetune \
    --n_memories_per_class -1 \
    --group final_cifar_experiments \
    --reduce_lr_on_plateau \
    --save_each_task_model \
    --use_best_model \
    --total_n_memories 100000

### Joint training
python ./prior_main.py \
    --method finetune \
    --n_memories_per_class -1 \
    --group final_cifar_experiments \
    --reduce_lr_on_plateau \
    --save_each_task_model \
    --complete_info \
    --use_best_model \
    --incremental_joint \
    --total_n_memories -
    --method finetune \
    --n_memories_per_class 20 \
    --reduce_lr_on_plateau \
    --group final_cifar_experiments \
    --save_each_task_model \
    --epochs_per_task 140 \
    --total_n_memories -1 \
    --use_best_model

### ER-infinite
python ./prior_main.py \
    --method finetune \
    --n_memories_per_class -1 \
    --group final_cifar_experiments \
    --reduce_lr_on_plateau \
    --save_each_task_model \
    --use_best_model \
    --total_n_memories 100000

### Joint training
python ./prior_main.py \
    --method finetune \
    --n_memories_per_class -1 \
    --group final_cifar_experiments \
    --reduce_lr_on_plateau \
    --save_each_task_model \
    --complete_info \
    --use_best_model \
    --incremental_joint \
    --total_n_memories -1

### iCaRL-CNN

python ./prior_main.py \
    --method icarl_cnn \
    --n_memories_per_class 20 \
    --group final_cifar_experiments \
    --reduce_lr_on_plateau \
    --save_each_task_model \
    --total_n_memories -1 \
    --epochs_per_task 140 \

### iCaRL-CNN + SPL
python ./prior_main_ptm_training_icarl.py \
    --method icarl_norm_ptm \
    --n_memories_per_class 20 \
    --reduce_lr_on_plateau \
    --group final_cifar_experiments \
    --save_each_task_model \
    --epochs_per_task 140 \
    --total_n_memories -1 \
    --patience 20 \
    --dataset_path ./cifar100

### iCaRL-CNN + SPL + Infer-HCV
python ./prior_test_ptm_training.py \
    --run_id 12345 \ ### replace 12345 with the real id from the previous running
    --method icarl_cnn_ptm \
    --n_memories_per_class 20 \
    --group final_cifar_experiments \
    --reduce_lr_on_plateau \
    --save_each_task_model \
    --epochs_per_task 140 \
    --total_n_memories -1 \
    --dataset_path ./cifar100