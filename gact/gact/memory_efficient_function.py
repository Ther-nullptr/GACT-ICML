import torch

'''
pack the memory efficient function into one file
'''

def per_block_quantization(x, input_shape, quantization_shape = 64, eps = 1e-10):
    # compress then save x
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape

    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()
    s = (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values) / 255
    r_min = x.min(dim=-1, keepdim=True).values
    z = - r_min / (s + eps) - 128
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-128, max=127)).to(torch.int8)

    # save the quantization state
    quant_state = (s, r_min, z)

    return x, quant_state


def per_block_dequantization(x, input_shape, quant_state, quantization_shape = 64):
    s, r_min, z = quant_state
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    # convert 
    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()
    # dequantize
    x = s * (x.to(torch.float32) - z)
    # then convert back to the original shape
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(input_shape)

    return x


def dct_compression(x, input_shape, dct_processor, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    shape_for_dct1d = input_shape[:-2] + (group_size_1, quantization_shape, input_shape[-1])
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(shape_for_dct1d) # [32, 2, 64, 768]
    x = torch.round(torch.clamp(dct_processor(x), -128, 127)).to(torch.int8)
    x = x.reshape(input_shape) # [32, 128, 768]

    return x


def jpeg_compression(x, input_shape, jpeg_processor, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    jpeg_size_1 = input_shape[-2] // 8
    jpeg_size_2 = input_shape[-1] // 8
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape).permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(-1, jpeg_size_1, 8, jpeg_size_2, 8).permute(0, 1, 3, 2, 4) # [32, 16, 8, 96, 8] -> [32, 16, 96, 8, 8]
    x = torch.round(torch.clamp(jpeg_processor(x), -128, 127)).to(torch.int8)
    x = x.permute(0, 1, 3, 2, 4) # [32, 16, 8, 96, 8] 
    x = x.reshape(input_shape) # [32, 128, 768]

    return x


def naive_adjustment(x, input_shape, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(input_shape)

    return x
