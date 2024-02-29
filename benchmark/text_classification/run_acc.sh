for task in rte
do
    python -u run_glue.py --model_name_or_path /home/yujin-wa20/projects/LoRA_microsoft/examples/NLU/roberta-base --task_name $task --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 10 --seed 42 --output_dir log/$task/ --gact --opt_level LDCT+75 
done

# mnli sst2
# --pad_to_max_length