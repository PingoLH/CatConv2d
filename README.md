# CatConv2d
Concat+Conv2d fused all-in-one CUDA kernel extension for Pytorch

- Optimized for Volta, 10% faster than torch concat+conv (batch size = 1)



~~~
    x = CatConv2d(x_list)
    
    # effectivly equal to
    x = torch.cat(x_list, 1)
    x = conv.forward(x)
~~~

## Features
1. Conv1x1 (GEMM) CUDA kernel
2. Conv3x3 Winograd CUDA kernel, fused with input/weight transform
3. Take python list of Tensor as input

## Limitations
1. Forward only
2. Maximum 16 Tensors for the input list
3. NCHW layout
4. Hardcoded zero padding (padding=1)
5. Optimized for small batch size

## TODO List
1. Backward path
2. SASS level optimization
3. TensorCore 16bit version

## Requirement
- Cuda 10.2

## Usage
~~~
python setup.py install

python test.py cuda
~~~
