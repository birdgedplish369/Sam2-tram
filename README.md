# Using Sam2 to Optimize the Tram Video Processing Workflow

## Sam2-Tram 环境搭建说明

本项目需要同时支持 **[Sam2](https://github.com/facebookresearch/sam2)** 和 **[Tram](https://github.com/yufu-wang/tram)**，请严格按照以下步骤进行环境配置。  
⚠️ 注意：**PyTorch 版本必须固定为 `2.5.1+cu124`，否则会导致依赖冲突或无法运行！**

---

## 1. 创建 Conda 环境
建议使用 Conda 进行环境隔离：

```bash
conda create -n sam2-tram python=3.10.18 -y
conda activate sam2-tram
```

## 2. 安装 PyTorch 2.5.1 (CUDA 12.4)
```bash
pip install torch==2.5.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
⚠️ 请勿修改此版本！ 后续库都基于该版本构建。


## 3. 安装 CUDA Toolkit（仅需编译/运行时支持）

如果本地没有 CUDA Toolkit，可以单独安装（匹配 12.4）：
```bash
conda install nvidia/label/cuda-12.4.131::cuda-toolkit -y
```

## 4. 安装 Sam2 和 Tram 及其依赖

参考官方安装说明（保持 PyTorch 版本不变）
