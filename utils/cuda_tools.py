import os
import torch
import pycuda
from pycuda import compiler
import pycuda.driver as drv


def print_cuda_device():
    drv.init()
    print("%d device(s) found." % drv.Device.count())

    for ordinal in range(drv.Device.count()):
        dev = drv.Device(ordinal)
        print(ordinal, dev.name())


if __name__ == "__main__":
    print_cuda_device()
