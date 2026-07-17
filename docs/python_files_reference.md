# 📂 Python 文件详解文档

> **项目名称：** fmi_ddpm (Denoising Diffusion Probabilistic Models)  
> **文档生成日期：** 2026-07-17  
> **框架：** PyTorch

---

## 📁 项目结构概览

```
fmi_ddpm/
├── setup.py                        # 项目安装与依赖配置
├── ddpm/                           # 核心库包
│   ├── __init__.py                 # 包初始化文件
│   ├── diffusion.py                # 高斯扩散模型（核心算法）
│   ├── ema.py                      # 指数移动平均（EMA）
│   ├── unet.py                     # UNet 噪声估计网络
│   ├── utils.py                    # 通用工具函数
│   └── script_utils.py             # 脚本辅助工具
└── scripts/                        # 可执行脚本
    ├── train_cifar.py              # CIFAR-10 训练脚本
    └── sample_images.py            # 图像采样/生成脚本
```

---

## 1. `setup.py` — 项目安装与依赖配置

**位置：** `D:\GitHub\fmi_ddpm\setup.py`  
**用途：** 定义 Python 包的元数据和依赖项，使项目可通过 `pip install` 安装。

### 核心内容

```python
setup(
    name="ddpm",
    py_modules=["ddpm"],
    install_requires=["torch", "torchvision", "einops", "wandb", "joblib"],
)
```

### 依赖说明

| 依赖包 | 用途 |
|--------|------|
| `torch` | PyTorch 深度学习框架 |
| `torchvision` | 数据集加载、图像变换工具 |
| `einops` | 张量维度操作库（本项目中未直接使用） |
| `wandb` | Weights & Biases 实验追踪与可视化 |
| `joblib` | 并行计算与序列化（本项目中未直接使用） |

---

## 2. `ddpm/__init__.py` — 包初始化文件

**位置：** `D:\GitHub\fmi_ddpm\ddpm\__init__.py`  
**用途：** Python 包初始化标记文件。

### 说明

该文件内容为空，仅用于告知 Python 解释器 `ddpm/` 目录是一个可导入的包（package），使其他模块可以通过 `from ddpm import xxx` 的方式引用包内子模块。

---

## 3. `ddpm/diffusion.py` — 高斯扩散模型（核心算法）

**位置：** `D:\GitHub\fmi_ddpm\ddpm\diffusion.py`  
**用途：** 实现 DDPM 的核心算法逻辑，包括扩散过程、去噪过程、损失计算、图像采样和噪声调度生成。

### 3.1 类：`GaussianDiffusion`

**继承：** `torch.nn.Module`  
**功能：** 高斯扩散模型的主模块，封装了前向扩散、反向去噪、损失计算和图像生成的完整流程。

#### 构造函数参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `nn.Module` | 用于估计扩散噪声的神经网络（UNet） |
| `img_size` | `tuple` | 图像尺寸 (H, W) |
| `img_channels` | `int` | 图像通道数 |
| `num_classes` | `int` | 类别数（用于类别条件生成，设为 0 表示无条件） |
| `betas` | `np.ndarray` | 扩散调度 β 值数组 |
| `loss_type` | `str` | 损失类型，`"l1"` 或 `"l2"` |
| `ema_decay` | `float` | EMA 衰减率，默认 0.9999 |
| `ema_start` | `int` | 开始 EMA 更新的步数，默认 5000 |
| `ema_update_rate` | `int` | EMA 更新频率，默认每步更新 |

#### 关键方法

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `update_ema()` | — | — | 更新 EMA 模型的权重 |
| `remove_noise(x, t, y)` | 当前噪声图像 `x`、时间步 `t`、类别标签 `y` | 去噪后的图像 | 执行一步反向去噪 |
| `sample(batch_size, device, y)` | 批量大小、设备、类别标签 | 生成的图像 Tensor | 从纯噪声开始，逐步去噪生成完整图像 |
| `sample_diffusion_sequence(...)` | 同上 | 扩散序列（每步的中间结果） | 生成扩散过程的完整序列（用于 GIF 制作） |
| `perturb_x(x, t, noise)` | 原始图像 `x`、时间步 `t`、噪声 | 加噪后的图像 | 前向扩散：对图像添加指定时间步的噪声 |
| `get_losses(x, t, y)` | 图像、时间步、类别标签 | 损失标量 | 计算噪声估计的 L1/L2 损失 |
| `forward(x, y)` | 输入图像、类别标签 | 损失标量 | 训练时的前向传播，随机采样时间步并计算损失 |

#### 算法流程

```
训练阶段：
  1. 随机采样时间步 t ∈ [0, T]
  2. 生成随机噪声 ε ~ N(0, I)
  3. 对输入图像 x 添加噪声：x_t = √ᾱ_t · x + √(1-ᾱ_t) · ε
  4. 用 UNet 预测噪声 ε_θ(x_t, t, y)
  5. 计算损失 L = MSE(ε_θ, ε) 或 L1(ε_θ, ε)

采样阶段：
  1. 从纯噪声 x_T ~ N(0, I) 开始
  2. 从 t = T-1 到 t = 0，逐步去噪：
     x_{t-1} = 去噪公式(x_t, t, y) + σ_t · z  (z ~ N(0,I))
  3. 输出生成的图像 x_0
```

### 3.2 函数：`generate_cosine_schedule(T, s=0.008)`

**功能：** 生成余弦噪声调度（Cosine Schedule）。

基于 Improved DDPM 论文中的余弦调度公式：

$$f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

$$\beta_t = \min\left(1 - \frac{f(t)}{f(t-1)}, 0.999\right)$$

| 参数 | 说明 |
|------|------|
| `T` | 总时间步数 |
| `s` | 余弦偏移参数，默认 0.008 |

### 3.3 函数：`generate_linear_schedule(T, low, high)`

**功能：** 生成线性噪声调度（Linear Schedule）。

使用 `np.linspace(low, high, T)` 生成从 `low` 到 `high` 的线性递增 β 值序列。

| 参数 | 说明 |
|------|------|
| `T` | 总时间步数 |
| `low` | β 的起始值 |
| `high` | β 的结束值 |

---

## 4. `ddpm/ema.py` — 指数移动平均（EMA）

**位置：** `D:\GitHub\fmi_ddpm\ddpm\ema.py`  
**用途：** 实现指数移动平均（Exponential Moving Average）机制，用于平滑模型权重，提升生成质量。

### 类：`EMA`

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__(decay)` | `decay`: 衰减率 | — | 初始化 EMA 衰减系数 |
| `update_average(old, new)` | `old`: 旧值, `new`: 新值 | 加权平均值 | 计算单次 EMA：`old × decay + new × (1 - decay)` |
| `update_model_average(ema_model, current_model)` | EMA 模型, 当前模型 | — | 遍历所有参数，逐个更新 EMA 模型的权重 |

#### 原理

EMA 通过历史权重的指数加权平均来稳定模型：

$$\theta_{EMA} = \theta_{EMA} \times \text{decay} + \theta_{current} \times (1 - \text{decay})$$

其中 `decay` 越接近 1，EMA 模型越平滑、变化越慢。这是扩散模型中提升生成质量的标准技术。

---

## 5. `ddpm/unet.py` — UNet 噪声估计网络

**位置：** `D:\GitHub\fmi_ddpm\ddpm\unet.py`  
**用途：** 实现用于估计扩散噪声的 UNet 架构网络，包含残差块、注意力机制、时间/类别条件注入等模块。

### 5.1 辅助函数：`get_norm(norm, num_channels, num_groups)`

**功能：** 根据指定类型返回对应的归一化层。

| `norm` 值 | 返回的归一化层 |
|-----------|--------------|
| `"in"` | `InstanceNorm2d`（带仿射变换） |
| `"bn"` | `BatchNorm2d` |
| `"gn"` | `GroupNorm` |
| `None` | `Identity`（无归一化） |

### 5.2 类：`PositionalEmbedding`

**功能：** 计算时间步的位置编码（Positional Encoding）。

采用 Transformer 风格的 sinusoidal 位置编码：

$$\text{emb}(t, 2i) = \sin(t \cdot \omega_i), \quad \text{emb}(t, 2i+1) = \cos(t \cdot \omega_i)$$

其中 $$\omega_i = \exp\left(-\frac{\log(10000)}{\text{dim}/2} \cdot i\right)$$

| 参数 | 说明 |
|------|------|
| `dim` | 编码维度（必须为偶数） |
| `scale` | 时间步的线性缩放系数，默认 1.0 |

### 5.3 类：`Downsample`

**功能：** 将特征图空间尺寸缩小 2 倍（使用步长为 2 的卷积）。

- 输入形状：`(N, C, H, W)`
- 输出形状：`(N, C, H//2, W//2)`

### 5.4 类：`Upsample`

**功能：** 将特征图空间尺寸放大 2 倍（使用最近邻插值 + 卷积，避免棋盘格伪影）。

- 输入形状：`(N, C, H, W)`
- 输出形状：`(N, C, H*2, W*2)`

### 5.5 类：`AttentionBlock`

**功能：** 应用 QKV 自注意力机制，带残差连接。

#### 计算流程

```
1. 输入 x → 归一化 → 1×1 卷积生成 Q, K, V
2. Q: (B, H*W, C)  K: (B, C, H*W)  V: (B, H*W, C)
3. 注意力分数 = softmax(Q @ K / √C)
4. 输出 = Attention @ V → 1×1 卷积 + 残差连接
```

| 参数 | 说明 |
|------|------|
| `in_channels` | 输入通道数 |
| `norm` | 归一化类型，默认 `"gn"` |
| `num_groups` | GroupNorm 的组数，默认 32 |

### 5.6 类：`ResidualBlock`

**功能：** 双卷积残差块，支持时间和类别条件注入（通过偏置相加方式）。

#### 结构

```
输入 x
  ↓
归一化1 → 激活 → 卷积1
  ↓
+ 时间偏置 (可选) → time_bias(激活(time_emb))
  ↓
+ 类别偏置 (可选) → class_bias(y)
  ↓
归一化2 → 激活 → Dropout → 卷积2
  ↓
+ 残差连接 (1×1 卷积或恒等映射)
  ↓
注意力块 (可选)
  ↓
输出
```

| 参数 | 说明 |
|------|------|
| `in_channels` | 输入通道数 |
| `out_channels` | 输出通道数 |
| `dropout` | Dropout 率 |
| `time_emb_dim` | 时间嵌入维度（None 表示不用时间条件） |
| `num_classes` | 类别数量（None 表示不用类别条件） |
| `activation` | 激活函数 |
| `norm` | 归一化类型 |
| `num_groups` | GroupNorm 组数 |
| `use_attention` | 是否在输出后接注意力块 |

### 5.7 类：`UNet`

**功能：** 完整的 UNet 架构，用于估计扩散过程中的噪声。

#### 架构结构

```
输入 (C, H, W)
  │
  ├── 初始卷积 → base_channels
  │
  ├── ↓ Downsampling 路径（编码器）
  │     每个 stage：num_res_blocks × ResidualBlock → 可选 Downsample
  │     所有中间特征图存入 skips 列表
  │
  ├── ↓ 中间层（Bottleneck）
  │     ResidualBlock(use_attention=True)
  │     ResidualBlock(use_attention=False)
  │
  ├── ↑ Upsampling 路径（解码器）
  │     每个 stage：(num_res_blocks + 1) × ResidualBlock → 可选 Upsample
  │     每次 ResidualBlock 前 concat 对应的 skip 特征
  │
  └── 输出层
        归一化 → 激活 → 3×3 卷积 → 输出 (img_channels, H, W)
```

#### 时间条件注入（Time Conditioning）

```
时间步 t → PositionalEmbedding → Linear → SiLU → Linear → time_emb
                                                        ↓
                                        加到每个 ResidualBlock 的中间特征上
```

#### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `img_channels` | 必填 | 输入/输出通道数 |
| `base_channels` | 必填 | 基础通道数（首次卷积后） |
| `channel_mults` | `(1, 2, 4, 8)` | 各 stage 通道数倍率 |
| `num_res_blocks` | 2 | 每个 stage 的残差块数量 |
| `time_emb_dim` | None | 时间嵌入维度 |
| `time_emb_scale` | 1.0 | 时间步缩放 |
| `num_classes` | None | 类别数（支持类别条件生成） |
| `dropout` | 0.1 | Dropout 率 |
| `attention_resolutions` | `()` | 在哪些相对分辨率处应用注意力 |
| `norm` | `"gn"` | 归一化类型 |
| `num_groups` | 32 | GroupNorm 组数 |
| `initial_pad` | 0 | 初始填充（用于非 2 的幂次尺寸） |

---

## 6. `ddpm/utils.py` — 通用工具函数

**位置：** `D:\GitHub\fmi_ddpm\ddpm\utils.py`  
**用途：** 提供辅助函数，用于张量维度对齐和广播。

### 函数：`extract(a, t, x_shape)`

**功能：** 从一维数组 `a` 中根据索引 `t` 提取值，并 reshape 以适配 `x_shape` 的维度。

#### 工作原理

```python
def extract(a, t, x_shape):
    b, *_ = t.shape                # 获取批量大小
    out = a.gather(-1, t)          # 从 a 中按 t 的值提取对应元素
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # reshape 为 (B, 1, 1, ...)
```

#### 使用场景

在扩散过程中，需要将一维的调度参数（如 β、α）广播到与图像张量相同的维度，以便进行逐元素运算。例如：

```python
# 将 sqrt_alphas_cumprod[t] 广播到 (B, 1, 1, 1) 以与 x (B, C, H, W) 相乘
extract(self.sqrt_alphas_cumprod, t, x.shape) * x
```

---

## 7. `ddpm/script_utils.py` — 脚本辅助工具

**位置：** `D:\GitHub\fmi_ddpm\ddpm\script_utils.py`  
**用途：** 提供训练和采样脚本中常用的辅助函数，包括参数解析、数据变换、模型构建等。

### 函数清单

| 函数 | 功能 |
|------|------|
| `cycle(dl)` | 创建无限循环的数据加载器迭代器 |
| `get_transform()` | 返回图像预处理变换：ToTensor + 将像素值从 [0,1] 映射到 [-1,1] |
| `str2bool(v)` | 字符串到布尔值的解析（支持 "yes/true/t/y/1" 和 "no/false/f/n/0"） |
| `add_dict_to_argparser(parser, default_dict)` | 将字典的键值对批量添加为 argparse 参数 |
| `diffusion_defaults()` | 返回扩散模型的默认超参数配置字典 |
| `get_diffusion_from_args(args)` | 根据 argparse 参数构建完整的 GaussianDiffusion 实例 |

### `diffusion_defaults()` 默认参数详解

```python
defaults = dict(
    # 扩散过程参数
    num_timesteps=1000,          # 扩散总步数 T
    schedule="linear",           # 噪声调度类型："linear" 或 "cosine"
    loss_type="l2",              # 损失函数："l1" 或 "l2"
    use_labels=False,            # 是否使用类别条件

    # UNet 架构参数
    base_channels=128,           # 基础通道数
    channel_mults=(1, 2, 2, 2),  # 通道倍率
    num_res_blocks=2,            # 每个 stage 残差块数
    time_emb_dim=128 * 4,        # 时间嵌入维度 = 512
    norm="gn",                   # 归一化：GroupNorm
    dropout=0.1,                 # Dropout 率
    activation="silu",           # 激活函数
    attention_resolutions=(1,),  # 在 1× 分辨率处应用注意力

    # EMA 参数
    ema_decay=0.9999,            # EMA 衰减率
    ema_update_rate=1,           # EMA 更新频率
)
```

### `get_diffusion_from_args(args)` 构建流程

```
1. 创建 UNet 模型
   ├── img_channels=3
   ├── num_classes=10 (use_labels=True) 或 None (use_labels=False)
   └── 使用 args 中的架构参数

2. 生成噪声调度
   ├── schedule="cosine" → generate_cosine_schedule(T)
   └── schedule="linear" → generate_linear_schedule(T, low, high)

3. 组装 GaussianDiffusion
   ├── 输入尺寸：(32, 32)
   ├── 通道数：3
   ├── 类别数：10
   └── EMA、损失类型等参数从 args 读取
```

---

## 8. `scripts/train_cifar.py` — CIFAR-10 训练脚本

**位置：** `D:\GitHub\fmi_ddpm\scripts\train_cifar.py`  
**用途：** 在 CIFAR-10 数据集上训练 DDPM 模型，支持 WandB 实验追踪和模型检查点保存。

### 训练流程

```
1. 解析命令行参数
2. 创建 GaussianDiffusion 模型 + Adam 优化器
3. （可选）加载预训练模型和优化器检查点
4. （可选）初始化 WandB 实验追踪
5. 加载 CIFAR-10 训练集和测试集
6. 训练循环：
   ├── 从训练集取一个 batch
   ├── 前向传播计算损失 loss = diffusion(x, y)
   ├── 反向传播 + 梯度更新
   ├── 更新 EMA 模型权重
   ├── 每 log_rate 步：
   │   ├── 计算测试集损失
   │   ├── 生成 10 张样本图像
   │   └── 记录到 WandB（test_loss, train_loss, samples）
   └── 每 checkpoint_rate 步：
       └── 保存模型权重和优化器状态
```

### 关键默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | `2e-4` | Adam 学习率 |
| `batch_size` | 128 | 训练批量大小 |
| `iterations` | 800,000 | 总训练迭代次数 |
| `log_to_wandb` | `True` | 是否启用 WandB 日志 |
| `log_rate` | 1000 | 日志记录频率（每 N 步） |
| `checkpoint_rate` | 1000 | 检查点保存频率（每 N 步） |
| `schedule_low` | `1e-4` | 线性调度 β 起始值 |
| `schedule_high` | `0.02` | 线性调度 β 结束值 |

### 数据预处理

- 使用 `script_utils.get_transform()`：`ToTensor` + 将像素值从 `[0, 1]` 线性映射到 `[-1, 1]`
- 训练集：`datasets.CIFAR10(root='./cifar_train', train=True)`
- 测试集：`datasets.CIFAR10(root='./cifar_test', train=False)`

### WandB 日志内容

| 指标 | 说明 |
|------|------|
| `test_loss` | 测试集平均损失 |
| `train_loss` | 训练集滑动平均损失 |
| `samples` | 生成的 10 张样本图像（带/不带类别标签） |

---

## 9. `scripts/sample_images.py` — 图像采样/生成脚本

**位置：** `D:\GitHub\fmi_ddpm\scripts\sample_images.py`  
**用途：** 加载训练好的 DDPM 模型权重，生成新的图像样本。

### 采样流程

```
1. 解析命令行参数
2. 创建 GaussianDiffusion 模型
3. 加载训练好的模型权重 (.pth 文件)
4. 采样生成图像：
   ├── 如果使用类别条件 (use_labels=True)：
   │   └── 对每个类别 (0-9)，各生成 num_images/10 张
   └── 如果不使用类别条件：
       └── 生成 num_images 张无条件样本
5. 将生成的图像保存到指定目录
```

### 图像后处理

生成的图像从 `[-1, 1]` 范围反归一化到 `[0, 1]`：

```python
image = ((samples[image_id] + 1) / 2).clip(0, 1)
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必填 | 训练好的模型权重文件路径 |
| `--save_dir` | 必填 | 生成图像的保存目录 |
| `num_images` | 10000 | 生成图像的总数量 |
| `device` | 自动检测 | 计算设备（CUDA 优先） |
| `use_labels` | `False` | 是否按类别条件生成 |

### 输出文件命名

- **无条件生成：** `{save_dir}/{image_id}.png`
- **按类别生成：** `{save_dir}/{label}-{image_id}.png`（如 `3-42.png` 表示类别 3 的第 42 张图）

---

## 🔗 模块依赖关系

```
setup.py
  └── ddpm (包)

ddpm/__init__.py
  └── (空，包标记)

ddpm/utils.py
  └── extract() ← 被 diffusion.py 调用

ddpm/ema.py
  └── EMA() ← 被 diffusion.py 调用

ddpm/unet.py
  ├── PositionalEmbedding()
  ├── Downsample() / Upsample()
  ├── AttentionBlock()
  ├── ResidualBlock()
  └── UNet() ← 被 script_utils.py 调用

ddpm/diffusion.py
  ├── GaussianDiffusion() ← 被 script_utils.py 调用
  ├── generate_cosine_schedule() ← 被 script_utils.py 调用
  └── generate_linear_schedule() ← 被 script_utils.py 调用

ddpm/script_utils.py
  ├── cycle() ← 被 train_cifar.py 调用
  ├── get_transform() ← 被 train_cifar.py 调用
  ├── diffusion_defaults() ← 被两个脚本调用
  └── get_diffusion_from_args() ← 被两个脚本调用

scripts/train_cifar.py
  └── 训练入口，使用以上所有模块

scripts/sample_images.py
  └── 采样入口，使用以上部分模块
```

---

## 📋 总结

| 文件 | 行数(约) | 核心职责 |
|------|---------|---------|
| `setup.py` | ~8 | 包安装配置 |
| `ddpm/__init__.py` | 0 | 包初始化标记 |
| `ddpm/utils.py` | ~5 | 张量广播辅助函数 |
| `ddpm/ema.py` | ~15 | 指数移动平均 |
| `ddpm/unet.py` | ~280 | UNet 网络架构（噪声估计器） |
| `ddpm/diffusion.py` | ~150 | DDPM 核心算法（扩散/去噪/采样） |
| `ddpm/script_utils.py` | ~80 | 脚本辅助工具（参数解析/模型构建） |
| `scripts/train_cifar.py` | ~120 | CIFAR-10 训练脚本 |
| `scripts/sample_images.py` | ~45 | 图像生成脚本 |

**项目总计：约 700 行核心代码**，是一个精简而完整的 DDPM 实现。
