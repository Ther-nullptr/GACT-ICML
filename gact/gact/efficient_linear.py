import torch
import bitsandbytes.functional as F
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor

class EfficientMemoryLinearFunc(torch.autograd.Function):
    # only suitable for batched matmul: (BxMxK) @ (KxR) -> (BxKxR) or (BxKxR) @ (RxN) -> (BxKxN)
    # and LoRA do not have bias
    @staticmethod
    def forward(ctx, x, w, b, use_bias, compress_type, jpeg_processor, dct_processor):
        #print(x.shape, w.shape)
        if use_bias:
            ctx.needs_inputs_grad = [x.requires_grad, w.requires_grad, b.requires_grad]
        else:
            ctx.needs_inputs_grad = [x.requires_grad, w.requires_grad]
        ctx.compress_type = compress_type
        ctx.use_bias = use_bias
        if use_bias:
            output = x @ w.transpose(0, 1) + b[None, ...] # TODO: what is the dimension of b?
        else:
            output = x @ w.transpose(0, 1)
        ctx.original_x = x
        # shape preparation for DCT
        input_shape = x.shape
        ctx.input_shape = input_shape
        group_size_1 = input_shape[-2] // 64
        group_size_2 = input_shape[-1] // 64

        # quantize the cached activation
        if compress_type == 'JPEG' or compress_type == 'DCT':
            if len(input_shape) < 3:
                x = x.unsqueeze(0)
            
            #! the order is wrong now
            x = x.view(-1, input_shape[-2] // 64, 64, input_shape[-1] // 64, 64)
            x = x.permute(0, 1, 3, 2, 4)
            x = x.reshape(-1, 64 * 64).contiguous()

            s = (x.max(dim=-1, keepdim=True).values - x.min(dim=-1, keepdim=True).values) / 255
            r_min = x.min(dim=-1, keepdim=True).values
            z = - r_min / s - 128
            x = torch.round(x / s + z).to(torch.int8)

            # save the quantization state
            ctx.quant_state = (s, r_min, z)

        # compress the cached activation
        if compress_type == 'JPEG':
            jpeg_size_1 = input_shape[-2] // 8
            jpeg_size_2 = input_shape[-1] // 8
            x = x.view(-1, group_size_1, group_size_2, 64, 64).permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
            x = x.view(-1, jpeg_size_1, 8, jpeg_size_2, 8).permute(0, 1, 3, 2, 4) # [32, 16, 8, 96, 8] -> [32, 16, 96, 8, 8]
            x = jpeg_processor(x).to(torch.int8).permute(0, 1, 3, 2, 4)

        elif compress_type == 'DCT':
            shape_for_dct1d = input_shape[:-2] + (group_size_1, 64, input_shape[-1])
            x = x.reshape(-1, group_size_1, group_size_2, 64, 64)
            x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
            x = x.reshape(shape_for_dct1d) # [32, 2, 64, 768]
            x = dct_processor(x).to(torch.int8)
            x = x.reshape(input_shape) # [32, 128, 768]

        # if the compress type is not JPEG or DCT, then the input will not be compressed(do nothing)
        
        ctx.save_for_backward(x, w)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        use_bias = ctx.use_bias
        x, w = ctx.saved_tensors
        s, r_min, z = ctx.quant_state
        input_shape = ctx.input_shape
        
        # dequantize the cached activation
        if ctx.compress_type == 'JPEG':
            pass
        
        elif ctx.compress_type == 'DCT':
            group_size_1 = input_shape[-2] // 64
            group_size_2 = input_shape[-1] // 64
            # convert 
            x = x.view(-1, group_size_1, 64, group_size_2, 64)
            x = x.permute(0, 1, 3, 2, 4)
            x = x.reshape(-1, 64 * 64).contiguous()
            # dequantize
            x = s * (x.to(torch.float16) - z)
            # then convert back to the original shape
            x = x.reshape(-1, group_size_1, group_size_2, 64, 64)
            x = x.permute(0, 1, 3, 2, 4) #! the order is right now, [32, 2, 64, 12, 64]
            x = x.reshape(input_shape)

        # print(f'original x: {ctx.original_x}')
        # print(f'dequantized x: {x}')

        grad_input = grad_weight = grad_bias = None
        if ctx.needs_inputs_grad[0]:
            grad_input = grad_output @ w
        if ctx.needs_inputs_grad[1]:
            grad_weight = grad_output.transpose(-2, -1) @ x
        if use_bias and ctx.needs_inputs_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None
        

class EfficientMemoryLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, compress_type: str = "JPEG", compress_quality: int = 50):
        super().__init__(in_features, out_features, bias)
        self.compress_type = compress_type
        self.compress_quality = compress_quality
        self.jpeg_processor = JPEGProcessor(quality=compress_quality)
        self.dct_processor = DCTProcessor(quality=compress_quality)
        
    def forward(self, input: torch.Tensor):
        return EfficientMemoryLinearFunc.apply(
            input, 
            self.weight, 
            self.bias, 
            self.bias != None, 
            self.compress_type,
            self.jpeg_processor,
            self.dct_processor
        )
    