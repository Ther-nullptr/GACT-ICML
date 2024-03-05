from collections import OrderedDict
import json
import torch
import numpy as np

def uniform_sample_ref(input, sample_cnt, add_dataptr=True):
    step = max(torch.numel(input) // sample_cnt, 1)
    key = []
    if add_dataptr:
        key.append(input.data_ptr())
    for i in range(min(sample_cnt, torch.numel(input))):
        idx = i * step
        key.append(input.view(-1)[idx].item())
    return key

def uniform_sample(input, sample_cnt, add_dataptr=True):
    num_elem = input.numel()
    sample_cnt = min(num_elem, sample_cnt)
    key = []
    if add_dataptr:
        key.append(input.data_ptr())
    key += input.ravel()[torch.arange(0, sample_cnt).to(torch.long) *
                          (num_elem // sample_cnt)].tolist()
    return key

def random_sample(input, sample_cnt, add_dataptr=True):
    num_elem = input.numel()
    rng_state = torch.get_rng_state()
    seed = input.dim()
    torch.manual_seed(seed)
    key = []
    if add_dataptr:
        key.append(input.data_ptr())

    key += input.view(-1)[torch.randint(0, num_elem, (sample_cnt,))].tolist()

    torch.set_rng_state(rng_state)
    return key

def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated

def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    
    for x in tensors:
        if type(x)== int:
            ret += 4
        elif x.dtype in [torch.long]:
            ret += np.prod(x.size()) * 8
        elif x.dtype in [torch.float32, torch.int]:
            ret += np.prod(x.size()) * 4
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8, torch.uint8]:
            ret += np.prod(x.size()) * 1
        else:
            print("[Error] unsupport datatype ", x.dtype)
            exit(0)

    return ret

def get_weight_memory(model):
    total_size = 0
    for name, param in model.named_parameters():
        total_size += compute_tensor_bytes(param)
    return total_size

def get_gradient_memory(model):
    total_size = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_size += compute_tensor_bytes(param.grad)
    return total_size

def get_optimizer_memory(optimizer):
    #! divide "weight" and "master weight". we can find this by record the data type
    #! without amp, the weight in optimizer is the same tensor in model param
    #! so ignore the weight in optimizer, just care the momentum buffer
    total_size = 0
    for group in optimizer.param_groups:
        for param in group['params']:
            state = optimizer.state[param]
            # only suit for Adam
            if 'exp_avg' in state and state['exp_avg'] is not None:
                momentum_value = state['exp_avg']
                total_size += compute_tensor_bytes(momentum_value)
            if 'exp_avg_sq' in state and state['exp_avg_sq'] is not None:
                momentum_sq_value = state['exp_avg_sq']
                total_size += compute_tensor_bytes(momentum_sq_value)
    return total_size

def empty_cache(ratio):
    if ratio is None:
        return
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if reserved > 0 and allocated / reserved < ratio:
        torch.cuda.empty_cache()

def get_dct_matrix(size: int):
  D = torch.zeros((size, size))
  n = torch.tensor(size, dtype=torch.bfloat16)
  for i in range(size):
    for j in range(size):
      if i == 0:
        D[i][j] = torch.sqrt(n) ** (-1)
      else:
        D[i][j] = torch.sqrt(2 / n) * torch.cos((2 * j + 1) * i * torch.pi / (2 * n))
  return D

def get_dqf_matrix(quality_factor, flatten=True, interpolation=1.):
  original_data = torch.tensor(
    [
      [16, 11, 10, 16, 24, 40, 51, 61],
      [12, 12, 14, 19, 26, 58, 60, 55],
      [14, 13, 16, 24, 40, 57, 69, 56],
      [14, 17, 22, 29, 51, 87, 80, 62],
      [18, 22, 37, 56, 68, 109, 103, 77],
      [24, 35, 55, 64, 81, 104, 113, 92],
      [49, 64, 78, 87, 103, 121, 120, 101],
      [72, 92, 95, 98, 112, 100, 103, 99]
    ]
  ).to(torch.float32)

  for i in range(8):
    for j in range(8):
      if quality_factor < 50:
        original_data[i][j] = torch.floor(original_data[i][j] * (50 / quality_factor))
      else:
        original_data[i][j] = torch.floor(original_data[i][j] * (2. - quality_factor / 50) + 0.5)

  if flatten == False:
    return original_data

  zigzag = []
  for i in range(0, 15):
    for j in range(8):
      if i - j >= 0 and i - j < 8:
        if i % 2 == 0:
          zigzag.append(original_data[i - j][j])
        else:
          zigzag.append(original_data[j][i - j])

  zigzag = torch.tensor(zigzag)
  # interpolation
  zigzag = torch.nn.functional.interpolate(zigzag.unsqueeze(0).unsqueeze(0), scale_factor=interpolation, mode='linear').squeeze()
  
  return zigzag

class GlobalExpRecorder:
    def __init__(self):
        self.val_dict = OrderedDict()

    def record(self, key, value, float_round=6):
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)
        if isinstance(value, (float, np.float32, np.float64)):
            value = round(value, float_round)

        self.val_dict[key] = value

    def dump(self, filename):
        with open(filename, "a") as fout:
            fout.write(json.dumps(self.val_dict) + '\n')
        print("Save exp results to %s" % filename)

    def clear(self):
        pass


exp_recorder = GlobalExpRecorder()
