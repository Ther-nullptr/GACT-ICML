# write a list
efficient_list=("gemm" "hadamard" "layernorm" "linear" "rmsnorm" "silu" "softmax")

for i in "${efficient_list[@]}"
do
    rm -rf efficient_${i}.py
    ln -s /home/yujin-wa20/projects/efficient_operators/efficient_backward_operators/efficient_${i}.py efficient_${i}.py
done