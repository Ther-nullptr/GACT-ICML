# quality=30
# quantization_shape=16
# for linear_mode in "NONE" "NAIVE" "DCT" "JPEG"
# do
#     for gelu_mode in "DCT" "JPEG"
#     do
#         python -u run_glue.py --model_name_or_path roberta-base --task_name rte --max_length 128 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --learning_rate 3e-4 --num_train_epochs 10 --seed 42 --output_dir log/$task/ --gact --opt_level LDCT+75 --pad_to_max_length --lora --lora-all-linear --linear-mode $linear_mode --gelu-mode $gelu_mode --layer-norm-mode $gelu_mode --softmax-mode $gelu_mode --gelu-quantization-shape  $quantization_shape --layer-norm-quantization-shape  $quantization_shape --softmax-quantization-shape  $quantization_shape --linear-quality $quality --gelu-quality $quality --layer-norm-quality $quality --softmax-quality $quality
#     done
# done

for linear_quality in 10 20 30
do 
    for layer_norm_mode in "NAIVE" "JPEG"
    do 
        python -u run_glue.py \
        --model_name_or_path roberta-base \
        --task_name rte \
        --max_length 128 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 128 \
        --learning_rate 3e-4 \
        --num_train_epochs 10 \
        --seed 42 \
        --output_dir log/rte/ \
        --pad_to_max_length \
        --use-fp4 \
        --lora \
        --lora-all-linear \
        --linear-mode JPEG \
        --gelu-mode JPEG \
        --gemm-mode JPEG \
        --layer-norm-mode $layer_norm_mode \
        --softmax-mode NAIVE \
        --linear-quality $linear_quality \
        --gelu-quality 30 \
        --layer-norm-quality 75 \
        --softmax-quality 30 \
        --gemm-quality 30
    done
done

# mnli sst2
# --pad_to_max_length





