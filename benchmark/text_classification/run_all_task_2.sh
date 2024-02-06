for task in qnli
do
  for level in L0
  do
    echo "level: $level"
    python -u run_glue.py --model_name_or_path /home/yujin-wa20/projects/GACT-ICML/model/roberta-large --task_name $task --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 1e-5 --num_train_epochs 10 --seed 42 --output_dir log/$task/finetune-$level-large/ --gact --opt_level $level
  done
done

# mnli sst2 