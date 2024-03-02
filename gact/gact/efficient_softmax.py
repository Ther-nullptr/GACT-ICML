import torch
import torch.nn as nn
import torch.nn.functional as F
from gact.dct_processor import DCTProcessor

quantize_size = 64

class EfficientMemorySoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, compress_type, jpeg_processor, dct_processor):
        result = F.softmax(input, dim=dim)
        x = result

        ctx.original_shape = input.shape
        # merge the 1st and 2nd dimension
        x = x.view(-1, input.shape[-2], input.shape[-1])

        ctx.needs_inputs_grad = x.requires_grad
        input_shape = x.shape
        ctx.input_shape = input_shape
        ctx.compress_type = compress_type
        group_size_1 = input_shape[-2] // quantize_size
        group_size_2 = input_shape[-1] // quantize_size

        x = x.view(-1, input_shape[-2] // quantize_size, quantize_size, input_shape[-1] // quantize_size, quantize_size)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(-1, quantize_size * quantize_size).contiguous()

        s = (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values) / 255
        r_min = x.min(dim=-1, keepdim=True).values
        z = - r_min / (s + 1e-10) - 128
        x = torch.round(torch.clamp(x / (s + 1e-10) + z, -128, 127)).to(torch.int8)

        # save the quantization state
        ctx.quant_state = (s, r_min, z)

        if compress_type == 'JPEG':
            jpeg_size_1 = input_shape[-2] // 8
            jpeg_size_2 = input_shape[-1] // 8
            x = x.view(-1, group_size_1, group_size_2, quantize_size, quantize_size).permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
            x = x.view(-1, jpeg_size_1, 8, jpeg_size_2, 8).permute(0, 1, 3, 2, 4) # [32, 16, 8, 96, 8] -> [32, 16, 96, 8, 8]
            x = jpeg_processor(x).to(torch.int8).permute(0, 1, 3, 2, 4)

        elif compress_type == 'DCT':
            shape_for_dct1d = input_shape[:-2] + (group_size_1, quantize_size, input_shape[-1])
            x = x.reshape(-1, group_size_1, group_size_2, quantize_size, quantize_size)
            x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
            x = x.reshape(shape_for_dct1d) # [32, 2, 64, 768]
            x = torch.round(torch.clamp(dct_processor(x), -128, 127)).to(torch.int8)
            x = x.reshape(input_shape) # [32, 128, 768]
  
        ctx.save_for_backward(x)
        ctx.dim = dim
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        dim = ctx.dim
        s, r_min, z = ctx.quant_state
        input_shape = ctx.input_shape

        # dequantize the cached activation
        if ctx.compress_type == 'JPEG':
          pass
        
        elif ctx.compress_type == 'DCT':
          group_size_1 = input_shape[-2] // quantize_size
          group_size_2 = input_shape[-1] // quantize_size
          # convert 
          x = x.view(-1, group_size_1, quantize_size, group_size_2, quantize_size)
          x = x.permute(0, 1, 3, 2, 4)
          x = x.reshape(-1, quantize_size * quantize_size).contiguous()
          # dequantize
          x = s * (x.to(torch.float16) - z)
          # then convert back to the original shape
          x = x.reshape(-1, group_size_1, group_size_2, quantize_size, quantize_size)
          x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
          x = x.reshape(input_shape)

        # demerge the 1st and 2nd dimension
        x = x.view(ctx.original_shape)
        grad_input = x * (grad_output - (grad_output * x).sum(dim=dim, keepdim=True))
        return grad_input, None, None, None, None

class EfficientMemorySoftmax(nn.Module):
    def __init__(self, dim, compress_type: str = "JPEG", compress_quality: int = 50):
        super(EfficientMemorySoftmax, self).__init__()
        self.dim = dim
        self.compress_type = compress_type
        self.compress_quality = compress_quality
        self.jpeg_processor = None # JPEGProcessor(quality=compress_quality)
        self.dct_processor = DCTProcessor(quality=compress_quality)

    def forward(self, input):
        return EfficientMemorySoftmaxFunc.apply(
          input, 
          self.dim,
          self.compress_type,
          self.jpeg_processor,
          self.dct_processor
        )