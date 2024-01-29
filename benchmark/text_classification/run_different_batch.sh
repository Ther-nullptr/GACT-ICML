# write a loop of 1,2,4,8,16,32,64,128

# 1. set up the environment

for batch_size in 1 2 4 8 16 32 64 128
do
    echo "-------------batch_size: $batch_size-------------"
    python -u run_glue.py --model_name_or_path roberta-base --task_name rte --max_length 128 --per_device_train_batch_size $batch_size --per_device_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 1 --seed 42 --pad_to_max_length  --output_dir log/rte/L1/ --gact --opt_level L1 --get-mem | grep "MB"
    echo "-------------------------------------------------"
done