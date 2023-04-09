import time
from functools import reduce

import numpy as np
from typing import Union
from multiprocessing import Pool

import torch


# 多参数用字典传递，写成这样
def gen_noise(args):
    args['size'].insert(0, args['num'])
    # return np.random.normal(loc=args['loc'], scale=args['scale'], size=args['size'])
    return torch.normal(args['loc'], args['scale'], size=args['size'])


def run_pool():
    cpu_worker_num = 4
    process_args = [{'num': 10000, 'loc': 0, 'scale': 1, 'size': [64, 64]}] * 7

    t1 = time.time()
    with Pool(cpu_worker_num) as p:
        outputs = p.map(gen_noise, process_args)
    t2 = time.time()
    print(t2 - t1)
    return outputs


# 主线程不建议写在 if外部。
if __name__ == '__main__':
    # a = torch.asarray([[1, 2, 3]])
    # b = torch.asarray([[2, 2, 2]])
    # c = torch.asarray([[3, 3, 3]])
    # d = [a, b, c]
    # w = torch.cat(d,dim=0)
    # print(w)
    a = torch.asarray([1.1,2.1,3])
    print(torch.round(a))

# t1 = time.time()
# # # a = torch.normal(0, 1, size=(70000, 224, 224))
# # a = np.random.normal(loc=0, scale=1, size=(35000, 64, 64)).astype('float32')
# # b = np.random.normal(loc=0, scale=1, size=(35000, 64, 64)).astype('float32')
# # c = np.concatenate((a,b),axis=0)
# x = np.zeros((70000, 64, 64), dtype='float32')
# x = np.random.normal(loc=0, scale=1, size=x.shape)
# t2 = time.time()
#
# print(t2 - t1)
