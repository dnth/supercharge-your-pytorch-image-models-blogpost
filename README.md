# Pytorch to ONNX-TensorRT

This repository contains a script to convert a PyTorch model to ONNX format and then to TensorRT format.

## Prerequisites

- PyTorch
- ONNX
- TensorRT

## Installation
For simplicity, I'll use a conda environment with Python 3.11.

Setup conda environment:
```bash
conda create -n pt-to-onnx-tensorrt python=3.11
conda activate pt-to-onnx-tensorrt
```


1. Install CUDA components:
   ```bash
   conda install -y -c nvidia cuda=12.2.2 cuda-tools=12.2.2 cuda-toolkit=12.2.2 cuda-version=12.2 cuda-command-line-tools=12.2.2 cuda-compiler=12.2.2 cuda-runtime=12.2.2
   ```

2. Install cuDNN:
   ```bash
   conda install cudnn==9.2.1.18
   ```

3. Install ONNX Runtime GPU:
   ```bash
   pip install -U onnxruntime-gpu==1.19.2
   ```
4. Install TensorRT:
   ```bash
   pip install tensorrt==10.1.0 tensorrt-cu12==10.1.0 tensorrt-cu12-bindings==10.1.0 tensorrt-cu12-libs==10.1.0
   ```

5. Install TIMM:
   ```bash
   pip install timm, onnx, cupy
   ```

6. Set up library paths:
   ```bash
   export LD_LIBRARY_PATH="/path/to/your/conda/env/lib:$LD_LIBRARY_PATH"
   export LD_LIBRARY_PATH="/path/to/your/conda/env/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH"
   ```
   Note: Adjust the paths according to your Conda environment location.


## Notebooks
Benchmark notebooks:
- [Benchmark TIMM](./notebooks/benchmark_timm.ipynb)
- [Benchmark ONNX Runtime CPU](./notebooks/benchmark_onnxruntime_cpu.ipynb)
- [Benchmark ONNX Runtime GPU](./notebooks/benchmark_onnxruntime_gpu.ipynb)
- [Benchmark TensorRT](./notebooks/benchmark_tensorrt.ipynb)


Conversion notebooks:
- [Pytorch to ONNX](./notebooks/pytorch_to_onnx.ipynb)
- [ONNX to TensorRT](./notebooks/onnx_to_tensorrt.ipynb)    
