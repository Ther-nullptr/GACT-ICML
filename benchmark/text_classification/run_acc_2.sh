for layers in 6 7 8 9 10 11
do
  for task in rte
  do
    for level in L0
    do
      python -u run_glue.py --model_name_or_path roberta-base --task_name $task --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 10 --seed 42 --output_dir log/$task/linear-probe/ --gact --opt_level $level --sparse-bp --sparse-bp-freeze-range "[i for i in range($layers)]" --sparse-bp-freeze-layer "['output.dense']"
    done
  done
done

# mnli sst2 