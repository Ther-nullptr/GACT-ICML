for level in L0 L1
do
  echo "level: $level"
  python -u run_glue.py --model_name_or_path roberta-base --task_name rte --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 10 --seed 42 --pad_to_max_length  --output_dir log/rte/LoRA-$level/ --gact --opt_level $level --optimizer-4bit --get-mem | grep "MB"
done