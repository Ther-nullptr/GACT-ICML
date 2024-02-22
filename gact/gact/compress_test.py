from gact.ops import *

if __name__ == '__main__':
  torch_fp16_data = torch.tensor(
    [
      [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ],
      [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
      ],
    ],
    dtype=torch.float16,
    device='cuda'
  )

  q_input, q_scale, q_min = no_scheme_quantize_pack(torch_fp16_data, 4, 0)
  print(q_input, q_scale, q_min)