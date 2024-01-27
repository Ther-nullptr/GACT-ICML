from torch import nn, Tensor, Size
from torch.types import Number
import typing

def flops_zero() -> int:
    return 0


def flops_elemwise(result_shape: Size) -> int:
    return result_shape.numel()


def flops_matmul(tensor1_shape: Size, tensor2_shape: Size, result_shape: Size) -> int:
    # ref: https://github.com/zhijian-liu/torchprofile/blob/6d80fe57bb8c6bc9f789da7925fac6547fa9502b/torchprofile/handlers.py#L35
    def get_reduce_dim_shape(_s: Size, is_first_mat: bool):
        return _s[0] if len(_s) == 1 else _s[-1 if is_first_mat else -2]

    reduce_dim_shape = get_reduce_dim_shape(tensor1_shape, True)
    assert reduce_dim_shape == get_reduce_dim_shape(tensor2_shape, False)
    return (2 * reduce_dim_shape - 1) * result_shape.numel()


def flops_convnd(module: nn.modules.conv._ConvNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    return (2 * kernel_size.numel() * module.in_channels - int(module.bias is None) * module.groups) * result_shape.numel()


def flops_avgpoolnd(module: nn.modules.pooling._AvgPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    return kernel_size.numel() * result_shape.numel()


def flops_adaptive_avgpoolnd(module: nn.modules.pooling._AdaptiveAvgPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(input_shape[2:], result_shape[2:])
    )
    return kernel_size.numel() * result_shape.numel()


def flops_maxpoolnd(module: nn.modules.pooling._AvgPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size([__k]) if isinstance(__k := module.kernel_size, int) else Size(__k)
    return (kernel_size.numel() - 1) * result_shape.numel()


def flops_adaptive_maxpoolnd(module: nn.modules.pooling._AdaptiveMaxPoolNd, input_shape: Size, result_shape: Size) -> int:
    kernel_size = Size(
        i_size // o_size if (i_size % o_size) == 0 else i_size - o_size * (i_size // o_size) + 1
        for i_size, o_size in zip(input_shape[2:], result_shape[2:])
    )
    return (kernel_size.numel() - 1) * result_shape.numel()


def ModuleFLOPs_zero(module: nn.Linear, result: Tensor, *args, **kwargs) -> int:
    return flops_zero()


def ModuleFLOPs_elemwise(module: nn.Module, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape  # [..., d_in]
    result_shape = result.shape
    assert input_shape == result_shape

    total_flops = flops_elemwise(result_shape)
    return total_flops


def ModuleFLOPs_Linear(module: nn.Linear, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape  # [..., d_in]
    weight_shape = module.weight.T.shape  # [d_out, d_in].T -> [d_in, d_out]
    result_shape = result.shape

    assert input_shape[-1] == weight_shape[0], f"{input_shape}, {weight_shape}"
    matmul_shape = Size(list(input_shape[:-1]) + list(weight_shape[-1:]))
    assert matmul_shape == result_shape

    total_flops = flops_matmul(input_shape, weight_shape, result_shape)
    if module.bias is not None:
        total_flops += flops_elemwise(result_shape)

    return total_flops


def ModuleFLOPs_QK(result: Tensor) -> int:
    assert isinstance(result, Tensor)
    input_shape = result.shape # len(result_shape) == 3 or 4. This is the shape of Q. K is the same shape.
    input_shape_2 = result.transpose(-2, -1).shape # This is the shape of K.T
    output_shape = Size(list(input_shape[:-1]) + [input_shape_2[-1]])
    total_flops = flops_matmul(input_shape, input_shape_2, output_shape)
    return total_flops


def ModuleFLOPs_OV(result: Tensor) -> int:
    assert isinstance(result, Tensor)
    input_shape_2 = result.shape # This is the shape of V
    input_shape = Size(list(input_shape_2[:-1]) + [input_shape_2[-2]])
    outputshape = input_shape_2
    total_flops = flops_matmul(input_shape, input_shape_2, outputshape)
    return total_flops


def ModuleFLOPs_ConvNd(module: typing.Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_convnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_AvgPoolNd(module: typing.Union[nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_avgpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_AdaptiveAvgPoolNd(module: typing.Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_adaptive_avgpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_MaxPoolNd(module: typing.Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_maxpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_AdaptiveMaxPoolNd(module: typing.Union[nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape
    result_shape = result.shape

    total_flops = flops_adaptive_maxpoolnd(module, input_shape, result_shape)
    return total_flops


def ModuleFLOPs_Norm(module: typing.Union[nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm], result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape  # [..., d_in]
    result_shape = result.shape
    assert input_shape == result_shape

    # (X-mean)/std
    total_flops = flops_elemwise(input_shape) * 2
    if (hasattr(module, 'affine') and module.affine) or (hasattr(module, 'elementwise_affine'), module.elementwise_affine):
        total_flops += flops_elemwise(input_shape) * 2

    return total_flops


def ModuleFLOPs_GELU(module: nn.GELU, result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 1
    assert isinstance(args[0], Tensor)
    assert isinstance(result, Tensor)

    input_shape = args[0].shape  # [..., d_in]
    result_shape = result.shape
    assert input_shape == result_shape

    total_flops = flops_elemwise(result_shape)

    return total_flops


def FunctionFLOPs_zero(result: Tensor, *args, **kwargs) -> int:
    return flops_zero()


def FunctionFLOPs_elemwise(result, *args, **kwargs) -> int:
    assert len(args) == 2, len(args)

    total_flops = None
    if isinstance(result, Number):
        total_flops = 1
    elif isinstance(result, Tensor):
        total_flops = flops_elemwise(result.shape)
    else:
        raise TypeError(type(result))

    return total_flops


def FunctionFLOPs_matmul(result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 2, len(args)
    tensor_A, tensor_B = args
    assert isinstance(tensor_A, Tensor) and isinstance(tensor_B, Tensor)

    total_flops = flops_matmul(tensor_A.shape, tensor_B.shape, result.shape)
    return total_flops


def FunctionFLOPs_linear(result: Tensor, *args, **kwargs) -> int:
    assert len(args) == 3, len(args)
    input, weight, bias = args
    assert isinstance(input, Tensor) and isinstance(weight, Tensor)

    total_flops = flops_matmul(input.shape, weight.T.shape, result.shape)
    if bias is not None:
        total_flops += flops_elemwise(result.shape)
    return total_flops


def MethodFLOPs_zero(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    return flops_zero()


def MethodFLOPs_elemwise(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    return flops_elemwise(result.shape)


def MethodFLOPs_sum(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    this_shape = self_obj.squeeze().shape
    result_shape = result.squeeze().shape

    total_flops = None
    if len(result_shape) == 0:
        total_flops = self_obj.numel() - 1
    else:
        kept_shape = list(this_shape)
        for s in result_shape:
            kept_shape.remove(s)
        kept_shape = Size(kept_shape)
        total_flops = kept_shape.numel() * (result_shape.numel() - 1)

    return total_flops


def MethodFLOPs_softmax(self_obj: Tensor, result: Tensor, *args_tail, **kwargs) -> int:
    this_shape = self_obj.shape
    result_shape = result.shape
    assert this_shape == result_shape

    exp_flops = flops_elemwise(this_shape)

    dim_reduce: int = args_tail[0] if args_tail else kwargs.get('dim')
    dims_kept = list(this_shape)
    dims_kept.pop(dim_reduce)
    dims_kept = Size(dims_kept)
    sum_flops = (this_shape[dim_reduce] - 1) * dims_kept.numel()

    div_flops = flops_elemwise(this_shape)

    total_flops = exp_flops + sum_flops + div_flops
    return total_flops


def MethodFLOPs_softmax_from_Q(result: Tensor, *args_tail, **kwargs) -> int:
    result_shape = result.shape
    exp_flops = flops_elemwise(result_shape)

    dim_reduce: int = -1 # TODO: view carefully
    dims_kept = list(result_shape)
    dims_kept.pop(dim_reduce)
    dims_kept = Size(dims_kept)
    sum_flops = (result_shape[dim_reduce] - 1) * dims_kept.numel()

    div_flops = flops_elemwise(result_shape)

    total_flops = exp_flops + sum_flops + div_flops
    return total_flops