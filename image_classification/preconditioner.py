import torch
import math
import time


def householder(src, tar):
    N = src.shape[0]
    v = tar - src
    v = v / v.norm()
    return torch.eye(N) - 2 * v.view(N, 1) @ v.view(1, N)


Qs = [[], [torch.ones(1), 1.0]]


def init(max_bs):
    for i in range(2, max_bs+1):
        e1 = torch.zeros(i)
        e1[0] = 1
        ones = torch.ones(i) / math.sqrt(i)
        H = householder(e1, ones)
        Hmax = H.abs().max()
        Qs.append([H, Hmax])


def get_transform(x):
    t = time.time()
    N = x.shape[0]
    x = x.view(N, -1)

    # mvec = x.abs().max(1)[0]
    mvec = x.max(1)[0] - x.min(1)[0]
    rank = (-mvec).argsort()
    values = mvec[rank]

    # Get block configurations
    num_zeros = 0
    total_values = values.sum()
    while True:
        num_zeros += 1
        total_values -= values[N - num_zeros]
        num = num_zeros * values[N - num_zeros - 1] / total_values
        if num >= 1:
            break

    num_nonzeros = N - num_zeros
    nums = (num_zeros * values / total_values)[:num_nonzeros]
    nums = torch.floor(torch.cumsum(nums, 0) + 1e-7).int()

    # Construct the matrix
    T = torch.zeros(N, N)
    all_s = torch.zeros(N)

    cnt = num_nonzeros
    indices = []
    index_cnt = 0
    print(time.time() - t)
    t = time.time()

    for i in range(num_nonzeros):
        # [i] + [cnt ~ num_nonzeros + nums[i]]
        indices.append(i)
        lambda_1 = values[i]
        lambda_2 = values[cnt]
        sz = num_nonzeros + nums[i] - cnt + 1
        Q, Qmax = Qs[sz]
        w = torch.tensor([lambda_1 / math.sqrt(sz), lambda_2 * Qmax])
        s = torch.tensor([w[0] ** (-1 / 3), (w[1] / (sz - 1)) ** (-1 / 3)])
        s *= (1 / s).norm()
        all_s[index_cnt] = s[0]
        all_s[index_cnt+1 : index_cnt+sz] = s[1]
        T[index_cnt:index_cnt+sz, index_cnt:index_cnt+sz] = Q
        index_cnt += sz
        for j in range(cnt, num_nonzeros + nums[i]):
            indices.append(j)
        cnt = num_nonzeros + nums[i]

    print(time.time() - t)
    t = time.time()

    T = T @ torch.diag(all_s)
    indices = rank[indices]
    inv_indices = torch.zeros(N, dtype=torch.int64)
    inv_indices[indices] = torch.arange(N)

    T = T[inv_indices]
    T = T[:, inv_indices]

    print(time.time() - t)
    return T
