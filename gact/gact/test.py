import torch
from utils import get_dct_matrix, get_dqf_matrix

if __name__ == '__main__':
  base_matrix = torch.ones(64)
  base_matrix_2 = torch.ones(64) * 2
  # concactenate the two matrices
  base_matrix = torch.stack([base_matrix, base_matrix_2], dim=0)
  dqf_matrix = get_dqf_matrix(50, flatten=True)
  print(base_matrix)
  print(dqf_matrix)
  print(base_matrix / dqf_matrix)

  x = torch.tensor(
    [
      [
        [1,2],
        [3,4]
      ],
      [
        [5,6],
        [7,8]
      ]
    ]
  )

  y = torch.tensor(
    [
      [1,2],
      [3,4]
    ],
  )

  print(x / y)