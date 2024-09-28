![PyTorch to ONNX-TensorRT](https://dicksonneoh.com/images/portfolio/supercharge_your_pytorch_image_models/post_image.png)

This repository contains code to optimize PyTorch image models using ONNX Runtime and TensorRT, achieving up to 8x faster inference speeds. Read the full blog post [here](https://dicksonneoh.com/portfolio/supercharge_your_pytorch_image_models/).


## Installation
Create and activate a conda environment:

```bash
conda create -n supercharge_timm_tensorrt python=3.11
conda activate supercharge_timm_tensorrt
```
 Install required packages:


```bash
pip install timm
pip install onnx
pip install onnxruntime-gpu==1.19.2
pip install cupy-cuda12x
pip install tensorrt==10.1.0 tensorrt-cu12==10.1.0 tensorrt-cu12-bindings==10.1.0 tensorrt-cu12-libs==10.1.0
```

Install CUDA dependencies:
```bash
conda install -c nvidia cuda=12.2.2 cuda-tools=12.2.2 cuda-toolkit=12.2.2 cuda-version=12.2 cuda-command-line-tools=12.2.2 cuda-compiler=12.2.2 cuda-runtime=12.2.2
```

Install cuDNN:
```bash
conda install cudnn==9.2.1.18
```

Set up library paths:
```bash
export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/supercharge_timm_tensorrt/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/supercharge_timm_tensorrt/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH"
```

## Running the code

The following codes correspond to the steps in the blog post.

### PyTorch latency benchmark:
   ```bash
   python 01_pytorch_latency_benchmark.py
   ```
Read more [here](https://dicksonneoh.com/portfolio/supercharge_your_pytorch_image_models//#-baseline-latency)

### Convert model to ONNX:
   ```bash
   python 02_convert_to_onnx.py
   ```
Read more [here](https://dicksonneoh.com/portfolio/supercharge_your_pytorch_image_models//#-convert-to-onnx)

### ONNX Runtime CPU inference:
   ```bash
   python 03_onnx_cpu_inference.py
   ```
Read more [here](https://dicksonneoh.com/portfolio/supercharge_your_pytorch_image_models//#-onnx-runtime-on-cpu)

### ONNX Runtime CUDA inference:
   ```bash
   python 04_onnx_cuda_inference.py
   ```
Read more [here](https://dicksonneoh.com/portfolio/supercharge_your_pytorch_image_models//#-onnx-runtime-on-cuda)

### ONNX Runtime TensorRT inference:
   ```bash
   python 05_onnx_trt_inference.py
   ```
Read more [here](https://dicksonneoh.com/portfolio/supercharge_your_pytorch_image_models//#-onnx-runtime-on-tensorrt)

### Export preprocessing to ONNX:
   ```bash
   python 06_export_preprocessing_onnx.py
   ```
Read more [here](https://dicksonneoh.com/portfolio/supercharge_your_pytorch_image_models//#-bake-pre-processing-into-onnx)

### Merge preprocessing and model ONNX:
   ```bash
   python 07_onnx_compose_merge.py
   ```
Read more [here](https://dicksonneoh.com/portfolio/supercharge_your_pytorch_image_models//#-bake-pre-processing-into-onnx)

### Run inference on merged model:
   ```bash
   python 08_inference_merged_model.py
   ```
Read more [here](https://dicksonneoh.com/portfolio/supercharge_your_pytorch_image_models//#-bake-pre-processing-into-onnx)



<!-- # Pytorch to ONNX-TensorRT

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
- [ONNX to TensorRT](./notebooks/onnx_to_tensorrt.ipynb)     -->
