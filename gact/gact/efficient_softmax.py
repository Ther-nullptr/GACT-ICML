import torch
import torch.nn as nn
import torch.nn.functional as F
from gact.jpeg_processor import JPEGProcessor
from gact.dct_processor import DCTProcessor

from gact.memory_efficient_function import per_block_quantization, per_block_dequantization, dct_compression, jpeg_compression

class EfficientMemorySoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, compress_type, jpeg_processor, dct_processor):
        result = F.softmax(input, dim=dim)
        x = result

        # info collection
        ctx.original_shape = input.shape
        # merge the 1st and 2nd dimension
        x = x.view(-1, input.shape[-2], input.shape[-1])
        input_shape = x.shape
        ctx.input_shape = input_shape
        ctx.compress_type = compress_type
        ctx.needs_inputs_grad = x.requires_grad
        ctx.dim = dim

        # quantization
        x, quant_state = per_block_quantization(x, input_shape)
        ctx.quant_state = quant_state

        # compression
        if compress_type == 'JPEG':
            x = jpeg_compression(x, input_shape, jpeg_processor)
        elif compress_type == 'DCT':
            x = dct_compression(x, input_shape, dct_processor)
  
        ctx.save_for_backward(x)
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        dim = ctx.dim
        quant_state = ctx.quant_state
        input_shape = ctx.input_shape

        grad_input = None
        if ctx.needs_inputs_grad:
            # dequantize the cached activation
            x = per_block_dequantization(x, input_shape, quant_state)

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
        self.jpeg_processor = JPEGProcessor(quality=compress_quality)
        self.dct_processor = DCTProcessor(quality=compress_quality)

    def forward(self, input):
        return EfficientMemorySoftmaxFunc.apply(
          input, 
          self.dim,
          self.compress_type,
          self.jpeg_processor,
          self.dct_processor
        )