import torch
import typing
import bitsandbytes as bnb
import bitsandbytes.functional as F

class MixedSparseGatedMLPFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x1: torch.Tensor, 
        w_gate: torch.Tensor, b_gate: torch.Tensor, w_gate_state: typing.Tuple, w_gate_lora_a: torch.Tensor, w_gate_lora_b: torch.Tensor, 
        w_up: torch.Tensor, b_up: torch.Tensor, w_up_state: typing.Tuple, w_up_lora_a: torch.Tensor, w_up_lora_b: torch.Tensor, 
        w_down: torch.Tensor, b_down: torch.Tensor, w_down_state: typing.Tuple, w_down_lora_a: torch.Tensor, w_down_lora_b: torch.Tensor, 
        activation_forward: str, sparsity_ratio: float, maintain_channels: int
    ):  
        # forward process: gate_proj
        w_gate_dequant = F.dequantize_4bit(w_gate, w_gate_state).to(x1.dtype).t()
        y1_main = x1 @ w_gate_dequant + b_gate if b_gate is not None else x1 @ w_gate_dequant
        y1_lora_a = x1 @ w_gate_lora_a
        y1_lora = y1_lora_a @ w_gate_lora_b
        y1 = y1_main + y1_lora

        # forward process: up_proj
        w_up_dequant = F.dequantize_4bit(w_up, w_up_state).to(x1.dtype).t()
        y2_main = x1 @ w_up_dequant + b_up if b_up is not None else x1 @ w_up_dequant
        y2_lora_a = x1 @ w_up_lora_a
        y2_lora = y2_lora_a @ w_up_lora_b
        y2 = y2_main + y2_lora

        # apply activation function
        if activation_forward == 'relu':
            xL = torch.relu(y1)
        elif activation_forward == 'silu':
            xL = torch.silu(y1)
        elif activation_forward == 'gelu':
            xL = torch.gelu(y1)

        # hadamard product
        xR = y2
        x3 = xL * xR

        # forward process: down_proj
        w_down_dequant = F.dequantize_4bit(w_down, w_down_state).to(x3.dtype).t()
        y3_main = x3 @ w_down_dequant + b_down if b_down is not None else x3 @ w_down_dequant
        y3_lora_a = x3 @ w_down_lora_a
        y3_lora = y3_lora_a @ w_down_lora_b
        y3 = y3_main + y3_lora

        # save: x1, y1_lora_a, y1(for soft activations), mask(for hard activations), x2, y2_lora_a
        mask = y1 < 0
        if activation_forward != 'relu':
            xL = torch.relu(xL)

        # the pruning of x3 etc. is not urgent, we can implement it in other place
        zero_count_per_channel = (x3 == 0).sum(dim=-2) # [bs, seq_len, hidden_dim] -> [bs, hidden_dim]
        actual_maintain_channel = min(int(sparsity_ratio * x3.size(-1)), maintain_channels)
        # record the top sparsity_ratio channels
        _, topk_indices = zero_count_per_channel.topk(actual_maintain_channel, dim=-1, largest=True)

        x3_save = torch.zeros((*x3.shape[:-1], actual_maintain_channel), dtype=x3.dtype)
        xL_save = torch.zeros((*xL.shape[:-1], actual_maintain_channel), dtype=xL.dtype)
        xR_save = torch.zeros((*xR.shape[:-1], actual_maintain_channel), dtype=xR.dtype)

        xR = xR * mask # the xR is sparse version for storage

        for i in range(len(topk_indices)):
            batch_idx = i
            col_indices = topk_indices[i]
            x3_save[i] = x3[batch_idx, :, col_indices]
            xL_save[i] = xL[batch_idx, :, col_indices]
            xR_save[i] = xR[batch_idx, :, col_indices]

        ctx.save_for_backward(x1, y1_lora_a, mask, xL_save, xR_save, y2_lora_a, x3_save, y3_lora_a, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b)
        ctx.quant_state = w_gate_state, w_up_state, w_down_state
        ctx.topk_indices = topk_indices
        ctx.x3_shape = x3.shape
        ctx.x3_device = x3.device

        return y3
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        topk_indices = ctx.topk_indices
        w_gate_state, w_up_state, w_down_state = ctx.quant_state
        x1, y1_lora_a, mask, xL_save, xR_save, y2_lora_a, x3_save, y3_lora_a, w_gate, b_gate, w_gate_lora_a, w_gate_lora_b, w_up, b_up, w_up_lora_a, w_up_lora_b, w_down, b_down, w_down_lora_a, w_down_lora_b = ctx.saved_tensors

        # convert the x3, xL, xR to the original shape
        x3 = torch.zeros(ctx.x3_shape, device=ctx.x3_device, dtype=x3_save.dtype)
        xL = torch.zeros(ctx.x3_shape, device=ctx.x3_device, dtype=xL_save.dtype)
        xR = torch.zeros(ctx.x3_shape, device=ctx.x3_device, dtype=xR_save.dtype)

        for i in range(x3_save.shape[0]):
            x3[i, :, topk_indices[i]] = x3_save[i]
            xL[i, :, topk_indices[i]] = xL_save[i]
            xR[i, :, topk_indices[i]] = xR_save[i]

        # down proj part
        # d L / d w_down_lora_a = x3.T @ d L / d y3 @ w_down_lora_b.T
        # TODO: x2 maybe sparse
        grad_w_down_lora_a = x3.T @ (grad_output @ w_down_lora_b.T)
        # d L / d w_down_lora_b = y3_lora_a.T @ d L / d y3
        grad_w_down_lora_b = y3_lora_a.T @ grad_output
        # d L / d x3 = d L / d y3 @ w_down.T + d L / d y3 @ w_down_lora_b.T @ w_down_lora_a.T
        w_down_dequant = F.dequantize_4bit(w_down, w_down_state).to(grad_output.dtype).t()
        grad_x3 = grad_output @ w_down_dequant.T + grad_output @ w_down_lora_b.T @ w_down_lora_a.T

        # hadamard product
        # # d L / d xL = d L / d x3 * xR
        # grad_xL = grad_x3 * xR
        # d L / d xR = d L / d x3 * xL
        grad_xR = grad_x3 * xL
        # activation part
        grad_y1 = grad_x3 * xR # notice that the xR there is not the original xR, but the sparse version (xR * mask)

        # up proj part
        # d L / d w_up_lora_a = x1.T @ d L / d xL @ w_up_lora_b.T
        grad_w_up_lora_a = xR.T @ (grad_xR @ w_up_lora_b.T)
        # d L / d w_up_lora_b = y1_lora_a.T @ d L / d xL
        grad_w_up_lora_b = y2_lora_a.T @ grad_xR
        # d L / d x1 = d L / d xR @ w_up.T + d L / d xR @ w_up_lora_b.T @ w_up_lora_a.T
        w_up_dequant = F.dequantize_4bit(w_up, w_up_state).to(grad_output.dtype).t()
        grad_x1 = grad_xR @ w_up_dequant.T + grad_xR @ w_up_lora_b.T @ w_up_lora_a.T

        # gate proj part
        # d L / d w_gate_lora_a = x1.T @ d L / d xL @ w_gate_lora_b.T
        grad_w_gate_lora_a = x1.T @ (grad_y1 @ w_gate_lora_b.T)
        # d L / d w_gate_lora_b = y1_lora_a.T @ d L / d xL
        grad_w_gate_lora_b = y1_lora_a.T @ grad_y1
        # d L / d x1 = d L / d xL @ w_gate.T + d L / d xL @ w_gate_lora_b.T @ w_gate_lora_a.T
        w_gate_dequant = F.dequantize_4bit(w_gate, w_gate_state).to(grad_output.dtype).t()
        grad_x1 += grad_y1 @ w_gate_dequant.T + grad_y1 @ w_gate_lora_b.T @ w_gate_lora_a.T

        return grad_x1, None, None, None, grad_w_gate_lora_a, grad_w_gate_lora_b, None, None, None, grad_w_up_lora_a, grad_w_up_lora_b, None, None, None, grad_w_down_lora_a, grad_w_down_lora_b, None, None, None
    

class MixedSparseGatedMLP(torch.nn.Module):
    def __init__(self, activation_forward='relu', sparsity_ratio=0.5, maintain_channels=10, bias=False):
        super(MixedSparseGatedMLP, self).__init__()
        self.activation_forward = activation_forward
        self.maintain_channels = maintain_channels
        self.sparsity_ratio = sparsity_ratio

    def forward(
        self, input: torch.Tensor, 
        gate_proj_base: bnb.nn.modules.Linear4bit, gate_proj_lora_a: torch.nn.Linear, gate_proj_lora_b: torch.nn.Linear,
        up_proj_base: bnb.nn.modules.Linear4bit, up_proj_lora_a: torch.nn.Linear, up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit, down_proj_lora_a: torch.nn.Linear, down_proj_lora_b: torch.nn.Linear
    ):
        return MixedSparseGatedMLPFunc.apply(
            input,
            gate_proj_base.weight,
            gate_proj_base.bias,
            gate_proj_base.weight.quant_state,
            gate_proj_lora_a.weight.T,
            gate_proj_lora_b.weight.T,
            ############################
            up_proj_base.weight,
            up_proj_base.bias,
            up_proj_base.weight.quant_state,
            up_proj_lora_a.weight.T,
            up_proj_lora_b.weight.T,
            ############################
            down_proj_base.weight,
            down_proj_base.bias,
            down_proj_base.weight.quant_state,
            down_proj_lora_a.weight.T,
            down_proj_lora_b.weight.T,
            ############################
            self.activation_forward,
            self.sparsity_ratio,
            self.maintain_channels
        )