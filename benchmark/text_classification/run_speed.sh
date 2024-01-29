for i in 1 2 3 4 5
do 
  echo "L0"
  python run_glue.py --model_name_or_path /home/yujin-wa20/projects/GACT-ICML/model/roberta-large --task_name rte --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 1 --seed 42 --pad_to_max_length  --output_dir log/rte/L0/ --gact --opt_level L0 --get-speed | grep IPS
done

for i in 1 2 3 4 5
do 
  echo "L1"
  python run_glue.py --model_name_or_path /home/yujin-wa20/projects/GACT-ICML/model/roberta-large --task_name rte --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 1 --seed 42 --pad_to_max_length  --output_dir log/rte/L1/ --gact --opt_level L1 --get-speed | grep IPS
done

for i in 1 2 3 4 5
do 
  echo "L0+ckpt"
  python run_glue.py --model_name_or_path /home/yujin-wa20/projects/GACT-ICML/model/roberta-large --task_name rte --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 1 --seed 42 --pad_to_max_length  --output_dir log/rte/L0/ --gact --opt_level L0 --get-speed  --ckpt | grep IPS
done

for i in 1 2 3 4 5
do 
  echo "L1+ckpt"
  python run_glue.py --model_name_or_path /home/yujin-wa20/projects/GACT-ICML/model/roberta-large --task_name rte --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 2e-5 --num_train_epochs 1 --seed 42 --pad_to_max_length  --output_dir log/rte/L1/ --gact --opt_level L1 --get-speed  --ckpt | grep IPS
done
