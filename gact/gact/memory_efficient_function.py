import torch

'''
pack the memory efficient function into one file
'''

def naive_quantization(x, eps = 1e-10):
    s = (x.max() - x.min()) / 255
    r_min = x.min(dim=-1, keepdim=True).values
    z = - r_min / (s + eps) - 128
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-128, max=127)).to(torch.int8)

    # save the quantization state
    quant_state = (s, r_min, z)
    return x, quant_state

def per_block_quantization(x, input_shape, quantization_shape = 64, eps = 1e-10, bit=8):
    # compress then save x
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape

    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()
    s = (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values) / (2 ** bit - 1) # 2**8-1
    r_min = x.min(dim=-1, keepdim=True).values
    z = - r_min / (s + eps) - (2 ** (bit - 1)) # 2**7
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-(2 ** (bit - 1)), max=(2 ** (bit - 1)) - 1)).to(torch.int8)

    # save the quantization state
    quant_state = (s, r_min, z)
    return x, quant_state


def per_block_quantization_fake(x, input_shape, quantization_shape, eps = 1e-10):
    # compress then save x
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    
    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()

    #! a tricky method: if the input has high sparsity, do not use z, thus quantized value will also has lots of zeros
    zero_ratio = (x == 0).sum() / x.numel()
    
    s = (x.max() - x.min()) / 255
    r_min = x.min()
    z = - r_min / (s + eps) - 128 #  if zero_ratio > 0.5 else 0
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-128, max=127)).to(torch.int8)

    # save the quantization state
    quant_state = (s, r_min, z)
    return x, quant_state


def per_block_quantization_4bit(x, input_shape, quantization_shape = 64, eps = 1e-10):
    # compress then save x
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape

    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()
    s = (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values) / 15
    r_min = x.min(dim=-1, keepdim=True).values
    z = - r_min / (s + eps) - 8
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-8, max=7)).to(torch.int8)

    # save the quantization state
    quant_state = (s, r_min, z)
    return x, quant_state


def per_block_quantization_4bit_fake(x, input_shape, quantization_shape = 64, eps = 1e-10):
    # compress then save x
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape

    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()
    s = (x.max() - x.min()) / 15
    r_min = x.min()
    z = - r_min / (s + eps) - 8
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-8, max=7)).to(torch.int8)

    # save the quantization state
    quant_state = (s, r_min, z)
    return x, quant_state


def per_block_quantization_3bit(x, input_shape, quantization_shape = 64, eps = 1e-10):
    # compress then save x
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape

    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()
    s = (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values) / 7
    r_min = x.min(dim=-1, keepdim=True).values
    z = - r_min / (s + eps) - 4
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-4, max=3)).to(torch.int8)

    # save the quantization state
    quant_state = (s, r_min, z)
    return x, quant_state


def per_block_quantization_3bit_fake(x, input_shape, quantization_shape = 64, eps = 1e-10):
    # compress then save x
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape

    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()
    s = (x.max() - x.min()) / 7
    r_min = x.min()
    z = - r_min / (s + eps) - 4
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-4, max=3)).to(torch.int8)

    # save the quantization state
    quant_state = (s, r_min, z)
    return x, quant_state


def per_block_quantization_2bit(x, input_shape, quantization_shape = 64, eps = 1e-10):
    # compress then save x
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape

    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()
    s = (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values) / 3
    r_min = x.min(dim=-1, keepdim=True).values
    z = - r_min / (s + eps) - 2
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-2, max=1)).to(torch.int8)

    # save the quantization state
    quant_state = (s, r_min, z)
    return x, quant_state


def per_block_quantization_2bit_fake(x, input_shape, quantization_shape = 64, eps = 1e-10):
    # compress then save x
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape

    x = x.view(-1, group_size_1, quantization_shape, group_size_2, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(-1, quantization_shape * quantization_shape).contiguous()
    s = (x.max() - x.min()) / 3
    r_min = x.min()
    z = - r_min / (s + eps) - 2
    x = torch.round(torch.clamp(x / (s + eps) + z, min=-2, max=1)).to(torch.int8)

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

# C = DP
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


def dct_compression_cpu(x, input_shape, dct_processor, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    shape_for_dct1d = input_shape[:-2] + (group_size_1, quantization_shape, input_shape[-1])
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(shape_for_dct1d) # [32, 2, 64, 768]
    x = torch.round(torch.clamp(dct_processor.forward_cpu(x), -128, 127)).to(torch.int8)
    x = x.reshape(input_shape) # [32, 128, 768]
    return x

# C = PD^T
def dct_compression_transpose(x, input_shape, dct_processor, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    shape_for_dct1d = input_shape[:-2] + (input_shape[-2], group_size_2, quantization_shape)
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(shape_for_dct1d) # [32, 128, 12, 64]
    x = torch.round(torch.clamp(dct_processor.forward_transpose(x), -128, 127)).to(torch.int8)
    x = x.reshape(input_shape) # [32, 128, 768]
    return x  


def dct_compression_transpose_cpu(x, input_shape, dct_processor, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    shape_for_dct1d = input_shape[:-2] + (input_shape[-2], group_size_2, quantization_shape)
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(shape_for_dct1d) # [32, 128, 12, 64]
    x = torch.round(torch.clamp(dct_processor.forward_transpose_cpu(x), -128, 127)).to(torch.int8)
    x = x.reshape(input_shape) # [32, 128, 768]
    return x  


def dct_compression_for_compress(x, input_shape, dct_processor, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    shape_for_dct1d = input_shape[:-2] + (group_size_1, quantization_shape, input_shape[-1])
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(shape_for_dct1d) # [32, 2, 64, 768]
    original_x = dct_processor.forward_dct_no_quant(x).to(torch.int8)
    x = dct_processor.forward_dct(x).to(torch.int8)
    return x, original_x


def dct_compression_transpose_for_compress(x, input_shape, dct_processor, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    shape_for_dct1d = input_shape[:-2] + (input_shape[-2], group_size_2, quantization_shape)
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(shape_for_dct1d) # [32, 128, 12, 64]
    original_x = dct_processor.forward_transpose_dct_no_quant(x).to(torch.int8)
    x = dct_processor.forward_transpose_dct(x).to(torch.int8) # [32, 128, 768]
    return x, original_x  


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


def jpeg_compression_cpu(x, input_shape, jpeg_processor, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    jpeg_size_1 = input_shape[-2] // 8
    jpeg_size_2 = input_shape[-1] // 8
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape).permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(-1, jpeg_size_1, 8, jpeg_size_2, 8).permute(0, 1, 3, 2, 4) # [32, 16, 8, 96, 8] -> [32, 16, 96, 8, 8]
    x = torch.round(torch.clamp(jpeg_processor.forward_cpu(x), -128, 127)).to(torch.int8)
    x = x.permute(0, 1, 3, 2, 4) # [32, 16, 8, 96, 8] 
    x = x.reshape(input_shape) # [32, 128, 768]
    return x


def jpeg_compression_for_compress(x, input_shape, jpeg_processor, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    jpeg_size_1 = input_shape[-2] // 8
    jpeg_size_2 = input_shape[-1] // 8
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape).permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(-1, jpeg_size_1, 8, jpeg_size_2, 8).permute(0, 1, 3, 2, 4) # [32, 16, 8, 96, 8] -> [32, 16, 96, 8, 8]
    # x = torch.round(torch.clamp(jpeg_processor(x), -128, 127)).to(torch.int8)
    original_x = jpeg_processor.forward_jpeg_no_quant(x).to(torch.int8)
    x = jpeg_processor.forward_jpeg(x).to(torch.int8)
    original_x = original_x.permute(0, 1, 3, 2, 4) # [32, 16, 8, 96, 8]
    original_x = original_x.reshape(input_shape) # [32, 128, 768]
    x = x.permute(0, 1, 3, 2, 4) # [32, 16, 8, 96, 8] 
    x = x.reshape(input_shape) # [32, 128, 768]
    return x, original_x


def naive_adjustment(x, input_shape, quantization_shape = 64):
    group_size_1 = input_shape[-2] // quantization_shape
    group_size_2 = input_shape[-1] // quantization_shape
    x = x.reshape(-1, group_size_1, group_size_2, quantization_shape, quantization_shape)
    x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
    x = x.reshape(input_shape)
    return x


def mask_process(x, input_shape):
    # extract the mask
    min_val = torch.min(x)
    mask = (x == min_val)
    x[x == min_val] = 0.

    diagonal_mask = torch.eye(input_shape[-1]).unsqueeze(0).unsqueeze(0).to(x.device)
    broadcasted_diagonal_mask = diagonal_mask.expand(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    diagonal_x = x * broadcasted_diagonal_mask
    x = x + x.mT - diagonal_x

    return x, mask