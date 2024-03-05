import torch
import torch.nn as nn
import torch.nn.functional as F
from gact.jpeg_processor import JPEGProcessor
from gact.dct_processor import DCTProcessor

from gact.memory_efficient_function import per_block_quantization, per_block_dequantization, dct_compression, jpeg_compression, naive_adjustment


class SoftmaxController():
    def __init__(self, compress_type: str = "JPEG", quantization_shape: int = 64, jpeg_processor: JPEGProcessor = None, dct_processor: DCTProcessor = None):
        self.compress_type = compress_type
        self.jpeg_processor = jpeg_processor
        self.dct_processor = dct_processor
        self.quantization_shape = quantization_shape
    
    def pack(self, x):
        self.original_shape = x.shape

        if self.compress_type != 'NONE':
            input_shape = x.shape
            self.input_shape = input_shape

            # quantization
            x, quant_state = per_block_quantization(x, input_shape, self.quantization_shape)
            self.quant_state = quant_state

            # compression
            if self.compress_type == 'JPEG':
                x = jpeg_compression(x, input_shape, self.jpeg_processor, self.quantization_shape)
            elif self.compress_type == 'DCT':
                x = dct_compression(x, input_shape, self.dct_processor, self.quantization_shape)
            elif self.compress_type == 'NAIVE':
                x = naive_adjustment(x, input_shape, self.quantization_shape)
        
        return x

    def unpack(self, x):
        if self.compress_type != 'NONE':
            x = per_block_dequantization(x, self.input_shape, self.quant_state, self.quantization_shape)
        x = x.reshape(self.original_shape)
        return x


class EfficientMemorySoftmax(nn.Module):
    def __init__(self, dim, compress_type: str = "JPEG", compress_quality: int = 50, quantization_shape: int = 64):
        super(EfficientMemorySoftmax, self).__init__()
        self.dim = dim
        self.compress_type = compress_type
        self.compress_quality = compress_quality
        self.jpeg_processor = JPEGProcessor(quality=compress_quality)
        self.dct_processor = DCTProcessor(quality=compress_quality, interpolation=quantization_shape / 64)
        self.quantization_shape = quantization_shape
        self.controller = SoftmaxController(compress_type, quantization_shape, self.jpeg_processor, self.dct_processor)

    def forward(self, input):
        with torch.autograd.graph.saved_tensors_hooks(self.controller.pack, self.controller.unpack):
            return F.softmax(input, self.dim)