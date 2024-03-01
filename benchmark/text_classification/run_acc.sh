for task in rte
do
    python -u run_glue.py --model_name_or_path roberta-base --task_name $task --max_length 128 --per_device_train_batch_size 10 --per_device_eval_batch_size 128 --learning_rate 3e-4 --num_train_epochs 10 --seed 42 --output_dir log/$task/ --gact --opt_level LDCT+75 --pad_to_max_length --lora --lora-all-linears
done

# mnli sst2
# --pad_to_max_length