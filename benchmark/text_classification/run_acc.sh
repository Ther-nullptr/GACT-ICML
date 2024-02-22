for task in rte
do
  for level in L1.4
  do
    python -u run_glue.py --model_name_or_path roberta-base --task_name $task --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 10 --seed 42 --output_dir log/$task/ --gact  --opt_level $level
  done
done

# mnli sst2 