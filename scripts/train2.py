# ==============================================================================
# train_cifar.py — CIFAR-10 数据集上训练 DDPM 模型的完整脚本
# ==============================================================================
# 项目名称:   fmi_ddpm (Denoising Diffusion Probabilistic Models)
# 脚本功能:   在 CIFAR-10 数据集上训练去噪扩散概率模型，
#             支持 WandB 实验追踪、模型检查点保存、EMA 权重平滑、
#             类别条件/无条件训练、中断恢复等高级功能
#
# 使用示例:
#   # 无条件训练（默认）
#   python scripts/train_cifar.py --project_name ddpm-cifar10
#
#   # 类别条件训练（使用 CIFAR-10 的 10 个类别标签）
#   python scripts/train_cifar.py --project_name ddpm-cifar10-label --use_labels True
#
#   # 从检查点恢复训练
#   python scripts/train_cifar.py --project_name ddpm-cifar10 \
#       --model_checkpoint path/to/model.pth \
#       --optim_checkpoint path/to/optim.pth
#
#   # 自定义超参数
#   python scripts/train_cifar.py --project_name ddpm-cifar10 \
#       --learning_rate 1e-4 --batch_size 64 --iterations 500000
#
# 框架:       PyTorch
# ==============================================================================

import argparse        # 命令行参数解析库
import datetime        # 日期时间处理，用于生成运行名称
import os

import torch           # PyTorch 深度学习框架核心
import wandb           # Weights & Biases：实验追踪、超参数记录、可视化平台

from torch.utils.data import DataLoader    # PyTorch 数据加载器，支持批量读取、多线程、打乱等
from torchvision import datasets           # TorchVision 内置数据集（CIFAR-10 等）
from ddpm import script_utils              # 本项目脚本辅助工具（模型构建、数据变换、参数解析等）


# ==============================================================================
# 主函数 — 训练流程的入口点
# ==============================================================================
def main():
    """
    训练流程主函数。完整流程如下：

    1. 解析命令行参数
    2. 构建 DDPM 扩散模型和优化器
    3. （可选）加载预训练的检查点
    4. （可选）初始化 WandB 实验追踪
    5. 加载 CIFAR-10 训练集和测试集
    6. 进入训练循环：
       a. 前向传播 → 计算噪声预测损失
       b. 反向传播 → 计算梯度
       c. 优化器更新 → 更新模型参数
       d. EMA 更新 → 平滑模型权重
       e. 定期评估 → 测试集验证 + 生成样本
       f. 定期保存 → 模型和优化器检查点
    """

    # -------------------------------------------------------------------------
    # 步骤 1: 解析命令行参数，确定计算设备
    # -------------------------------------------------------------------------
    # create_argparser() 返回一个配置好的 argparse.ArgumentParser 对象，
    # .parse_args() 解析 sys.argv 中的命令行参数，返回 Namespace 对象
    # activation = {str} 'silu'
    # attention_resolutions = {tuple: 1} (1,)
    # base_channels = {int} 128
    # batch_size = {int} 128
    # channel_mults = {tuple: 4} (1, 2, 2, 2)
    # checkpoint_rate = {int} 1000
    # device = {device} device(type='cpu')
    # dropout = {float} 0.1
    # ema_decay = {float} 0.9999
    # ema_update_rate = {int} 1
    # iterations = {int} 800000
    # learning_rate = {float} 0.0002
    # log_dir = {str} 'D:\\GitHub\\fmi_ddpm\\ddpm_logs'
    # log_rate = {int} 1000
    # log_to_wandb = {bool} True
    # loss_type = {str} 'l2'
    # model_checkpoint = {NoneType} None
    # norm = {str} 'gn'
    # num_res_blocks = {int} 2
    # num_timesteps = {int} 1000
    # optim_checkpoint = {NoneType} None
    # project_name = {str} 'DDPM_CIFAR_TEST'
    # run_name = {str} 'ddpm-2026-07-17-18-18'
    # schedule = {str} 'linear'
    # schedule_high = {float} 0.02
    # schedule_low = {float} 0.0001
    # time_emb_dim = {int} 512
    # use_labels = {bool} False
    args = create_argparser().parse_args()

    # 确定计算设备：优先使用 GPU (CUDA)，如果没有可用 GPU 则回退到 CPU
    # torch.device 对象用于指定张量和模型所在的硬件设备
    device = args.device

    # -------------------------------------------------------------------------
    # 步骤 2: 构建模型和优化器
    # -------------------------------------------------------------------------
    try:
        # 调用 script_utils.get_diffusion_from_args(args) 根据命令行参数
        # 自动构建完整的 DDPM 训练系统，包括：
        #   - UNet 噪声估计网络（架构由 args.base_channels, args.channel_mults 等控制）
        #   - GaussianDiffusion 扩散模型（调度由 args.schedule, args.num_timesteps 控制）
        #   - EMA（指数移动平均）模型副本
        # 然后通过 .to(device) 将整个模型转移到目标设备（GPU/CPU）
        diffusion = script_utils.get_diffusion_from_args(args).to(device)


        # 创建 Adam 优化器
        #   diffusion.parameters(): 获取模型中所有可学习参数（UNet 的权重、偏置等）
        #   lr=args.learning_rate: 学习率，默认 2e-4（DDPM 原始论文推荐值）
        # Adam 优化器结合了动量（Momentum）和自适应学习率（RMSProp）的优点，
        # 是扩散模型训练中最常用的优化器
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        # -------------------------------------------------------------------------
        # 步骤 3: （可选）加载预训练的检查点以继续训练
        # -------------------------------------------------------------------------
        # model_checkpoint: 之前保存的模型权重文件路径
        # 如果提供了该参数，将加载的权重应用到当前模型
        # 典型场景：训练中断后从上次检查点恢复，或使用预训练模型进行微调
        if args.model_checkpoint is not None:
            # torch.load() 从 .pth 文件加载 state_dict（有序字典，键为参数名，值为 Tensor）
            # load_state_dict() 将加载的权重填充到模型的对应参数中
            diffusion.load_state_dict(torch.load(args.model_checkpoint))

        # optim_checkpoint: 之前保存的优化器状态文件路径
        # 加载优化器状态可以恢复 Adam 的动量（momentum）和二阶矩（v），
        # 确保训练从检查点继续时优化器的内部状态一致
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        # -------------------------------------------------------------------------
        # 步骤 4: （可选）初始化 WandB 实验追踪
        # -------------------------------------------------------------------------
        if args.log_to_wandb:
            # 如果启用了 WandB 日志但没有指定项目名称，抛出错误
            # project_name 是 WandB 上项目的唯一标识符，用于组织实验
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            # wandb.init(): 初始化一个新的 WandB 实验运行（Run）
            # 每次调用会创建一个新的实验记录，包含超参数配置、指标、图表等
            run = wandb.init(
                project=args.project_name,          # 项目名称，用于在 WandB 网站上分组
                entity='mapblue19-china-university-of-petroleum',                 # WandB 团队/组织名称（用户名或团队名）
                config=vars(args),                   # vars(args) 将 Namespace 转为字典，记录所有超参数
                name=args.run_name,                  # 本次运行的自定义名称，默认格式 "ddpm-YYYY-MM-DD-HH-MM"
            )

            # wandb.watch(): 监控模型的梯度范数、参数分布等
            # 这会在 WandB 面板上显示梯度直方图，帮助诊断训练问题（如梯度爆炸/消失）
            wandb.watch(diffusion)

        # -------------------------------------------------------------------------
        # 步骤 5: 加载 CIFAR-10 数据集
        # -------------------------------------------------------------------------
        # 从 args 中读取批量大小，控制每次迭代处理多少张图像
        # batch_size 越大，梯度估计越稳定，但需要更多 GPU 显存
        batch_size = args.batch_size

        # 创建 CIFAR-10 训练数据集对象
        # CIFAR-10 包含 50,000 张 32×32 彩色训练图像和 10,000 张测试图像
        # 分为 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
        train_dataset = datasets.CIFAR10(
            root='./cifar_train/',       # 数据集下载/缓存的本地根目录
            train=True,                 # True 表示加载训练集（50,000 张）
            download=True,              # 如果本地不存在，自动从官网下载
            transform=script_utils.get_transform(),  # 数据预处理变换
        )
        # script_utils.get_transform() 返回的组合变换：
        #   1. ToTensor(): 将 PIL 图像 [0,255] 转为 Tensor [0,1]，形状 (C, H, W)
        #   2. RescaleChannels(): 将像素值从 [0, 1] 线性映射到 [-1, 1]，公式: 2*x - 1
        #      原因：DDPM 使用 tanh 激活和归一化输入，[-1,1] 范围更适合扩散过程

        # 创建 CIFAR-10 测试数据集对象（用于定期验证）
        test_dataset = datasets.CIFAR10(
            root='./cifar_test',        # 测试集缓存目录（与训练集分开存储）
            train=False,                # False 表示加载测试集（10,000 张）
            download=True,              # 如果本地不存在，自动下载
            transform=script_utils.get_transform(),  # 使用与训练集相同的预处理
        )

        # 创建训练集数据加载器
        # script_utils.cycle() 将 DataLoader 包装为无限循环迭代器，
        # 当遍历完所有数据后自动重新开始，适配扩散模型需要大量迭代的特点
        # 这样就不需要在训练循环中手动处理 epoch 边界
        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,       # 每批处理的图像数量
            shuffle=True,                # 每个 epoch 开始时打乱数据顺序，防止模型记住数据顺序
            drop_last=True,              # 丢弃最后不足一个 batch 的数据，保证所有 batch 大小一致
            num_workers=2,               # 使用 2 个子进程并行加载数据，避免数据加载成为瓶颈
        ))

        # 创建测试集数据加载器
        # 测试集不需要 shuffle（顺序不影响验证结果）和 cycle（只需遍历一次）
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=2)

        # 累计训练损失（用于 log_rate 周期内的平均损失计算）
        # 初始化为 0，每步训练后累加，到达 log_rate 时计算平均值并重置
        acc_train_loss = 0

        # -------------------------------------------------------------------------
        # 步骤 6: 训练主循环 — 核心训练过程
        # -------------------------------------------------------------------------
        # 从 iteration=1 开始，到 args.iterations（默认 800,000）结束
        # 对于 CIFAR-10（50,000 张训练图像，batch_size=128），每个 epoch 约 390 步
        # 800,000 步 ≈ 2051 个 epoch，足够模型充分收敛
        for iteration in range(1, args.iterations + 1):

            # --- 6a. 设置模型为训练模式 ---
            # diffusion.train() 通知模型内部所有层（特别是 Dropout、BatchNorm 等）
            # 切换到训练行为模式：
            #   - Dropout 层：随机丢弃部分神经元（防止过拟合）
            #   - BatchNorm：使用当前 batch 的均值和方差
            #   - 启用梯度计算
            diffusion.train()

            # --- 6b. 从训练加载器获取下一个 batch ---
            # next(train_loader) 返回 (images, labels)，每个 Tensor 的 shape 为 (batch_size, ...)
            # x: 图像 Tensor，shape = (batch_size, 3, 32, 32)，值域 [-1, 1]
            # y: 类别标签 Tensor，shape = (batch_size,)，值域 [0, 9]（CIFAR-10 有 10 个类别）
            # 注意：train_loader 被 cycle() 包装，所以永远不会抛出 StopIteration
            x, y = next(train_loader)

            # 将图像和标签从 CPU 转移到目标设备（GPU/CPU）
            # 只有转移到设备上的张量才能参与模型的前向/反向计算
            x = x.to(device)
            y = y.to(device)

            # --- 6c. 前向传播 — 计算扩散损失 ---
            # 根据是否使用类别条件，决定传递给模型的参数
            # diffusion(x, y) 内部执行：
            #   1. 随机采样时间步 t ~ Uniform(0, T)（T 为总时间步数，默认 1000）
            #   2. 生成随机噪声 ε ~ N(0, I)，shape 与 x 相同
            #   3. 根据扩散公式对 x 加噪：x_t = √ᾱ_t · x + √(1-ᾱ_t) · ε
            #   4. 将 x_t、t（和 y）送入 UNet，预测噪声 ε_θ(x_t, t, y)
            #   5. 计算损失：L = MSE(ε_θ, ε) 或 L1(ε_θ, ε)
            # 返回值 loss 是一个标量 Tensor，表示当前 batch 的平均噪声预测误差
            if args.use_labels:
                # 类别条件训练：传入标签 y，UNet 通过嵌入层学习类别特定的生成模式
                # 适用于需要生成指定类别的图像
                loss = diffusion(x, y)
            else:
                # 无条件训练：不传入标签，模型学习整体数据分布
                # 生成的图像类别是随机的
                loss = diffusion(x)

            # 将损失值从 GPU Tensor 转为 Python 标量，并累加到累计损失中
            # .item() 从单元素 Tensor 中提取 Python 数字，避免 GPU 内存泄漏
            # acc_train_loss 用于计算 log_rate 周期内的平均训练损失
            acc_train_loss += loss.item()

            # --- 6d. 反向传播 — 计算梯度 ---
            # optimizer.zero_grad(): 将所有参数的梯度清零
            # 必须在每次 backward() 之前调用，否则梯度会累积（累加到上一次的梯度上）
            # 原因：PyTorch 默认累加梯度而非覆盖，这是为了支持梯度累积等高级用法
            optimizer.zero_grad()

            # loss.backward(): 从 loss 标量开始，通过计算图反向传播计算每个参数的梯度
            loss.backward()

            # optimizer.step(): 根据计算出的梯度更新模型参数
            # Adam 更新规则（简化版）：
            #   m_t = β1 * m_{t-1} + (1 - β1) * g_t          （一阶矩估计）
            #   v_t = β2 * v_{t-1} + (1 - β2) * g_t²          （二阶矩估计）
            #   θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)        （参数更新）
            # 其中 m̂_t 和 v̂_t 是偏差校正后的矩估计
            optimizer.step()

            # --- 6e. 更新 EMA（指数移动平均）模型权重 ---
            # EMA 是扩散模型中提升生成质量的关键技术
            # diffusion.update_ema() 内部逻辑：
            #   1. step += 1（内部计数器递增）
            #   2. 如果 step % ema_update_rate == 0：
            #      a. 如果 step < ema_start（默认 2000）：直接将当前模型权重复制到 EMA 模型
            #         （训练初期的权重不稳定，直接复制比加权平均更好）
            #      b. 否则：执行 EMA 更新
            #         θ_EMA = θ_EMA × decay + θ_current × (1 - decay)
            #         默认 decay = 0.9999，意味着 EMA 模型变化非常缓慢，起到平滑作用
            # EMA 模型不参与训练，仅用于采样（推理）阶段，能生成质量更高的图像
            diffusion.update_ema()

            # -------------------------------------------------------------------------
            # 步骤 6f: 定期评估 — 每 log_rate 步执行一次验证和采样
            # -------------------------------------------------------------------------
            # 检查当前迭代是否是日志记录周期的倍数
            # log_rate 默认为 1000，即每训练 1000 步进行一次评估
            # 这个频率需要在信息量和计算开销之间取得平衡
            if iteration % args.log_rate == 0:

                # --- 初始化测试损失累加器 ---
                test_loss = 0

                # --- 禁用梯度计算进行验证 ---
                # torch.no_grad() 上下文管理器：在此块内的所有操作不构建计算图，不记录梯度
                # 好处：
                #   1. 节省 GPU 显存（不需要存储中间激活值用于反向传播）
                #   2. 加速计算（跳过梯度相关的内存分配和操作）
                #   3. 防止在评估过程中意外修改模型参数
                with torch.no_grad():
                    # 设置模型为评估模式
                    # diffusion.eval() 与 diffusion.train() 相对：
                    #   - Dropout 层：不再随机丢弃，使用全部神经元
                    #   - BatchNorm：使用训练期间累积的全局均值和方差
                    #   - 但仍允许 forward 计算（与 model.eval() 行为一致）
                    diffusion.eval()

                    # 遍历整个测试集，计算平均测试损失
                    # test_loader 不是 cycle 包装的，所以会正常遍历完所有 batch 后退出循环
                    for x, y in test_loader:
                        # 将测试 batch 转移到目标设备
                        x = x.to(device)
                        y = y.to(device)

                        # 计算当前 batch 的损失（与训练时相同的前向传播逻辑）
                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        # 累加到测试损失总和
                        test_loss += loss.item()

                # --- 生成样本图像用于可视化 ---
                # diffusion.sample() 从纯噪声开始，逐步去噪生成新图像
                # 内部流程（反向扩散过程）：
                #   1. 生成纯噪声 x_T ~ N(0, I)，shape = (batch_size, 3, 32, 32)
                #   2. 从 t = T-1 递减到 t = 0，每一步：
                #      a. 使用 EMA 模型预测噪声 ε_θ(x_t, t, y)
                #      b. 根据 DDPM 反向公式计算 x_{t-1}
                #      c. 如果 t > 0，添加少量随机噪声 σ_t · z（z ~ N(0,I)）
                #   3. 返回最终的 x_0（生成的清晰图像）
                # 生成 10 张样本图像用于 WandB 可视化
                if args.use_labels:
                    # 类别条件采样：生成 10 张图像，每张对应一个类别（0-9）
                    # torch.arange(10, device=device) 生成 Tensor([0, 1, 2, ..., 9])
                    # 这样每个类别各生成一张样本，可以直观检查各类别的生成质量
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    # 无条件采样：生成 10 张随机类别的图像
                    samples = diffusion.sample(10, device)

                # --- 图像后处理：将 Tensor 转为可视化的 NumPy 数组 ---
                # 原始 samples 的值域是 [-1, 1]（训练时的输入范围）
                # 需要转换回 [0, 1] 范围才能正确显示为图像
                # 转换步骤：
                #   1. (samples + 1) / 2:  线性映射 [-1, 1] → [0, 1]
                #   2. .clip(0, 1):        截断超出 [0, 1] 范围的值（防止数值误差导致的越界）
                #   3. .permute(0, 2, 3, 1): 改变维度顺序 (B, C, H, W) → (B, H, W, C)
                #                          WandB Image 需要 (H, W, C) 格式的 NumPy 数组
                #   4. .numpy():           将 PyTorch Tensor 转为 NumPy 数组
                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()

                # --- 计算平均损失 ---
                # 将累加的总损失除以 batch 数量，得到平均损失
                # len(test_loader) = 测试集总样本数 // batch_size（因为 drop_last=True）
                test_loss /= len(test_loader)
                # acc_train_loss 是 log_rate 步的总损失，除以 log_rate 得到平均训练损失
                acc_train_loss /= args.log_rate

                # --- 将指标和样本记录到 WandB ---
                # wandb.log() 将键值对发送到 WandB 服务器，在网页面板上实时更新
                wandb.log({
                    "test_loss": test_loss,         # 测试集平均损失（越低越好，反映泛化能力）
                    "train_loss": acc_train_loss,   # 训练集平均损失（越低越好，反映拟合程度）
                    # samples 中的每张图像会被 WandB 作为独立的图片展示
                    # wandb.Image(sample) 将 NumPy 数组封装为 WandB 可渲染的图像对象
                    "samples": [wandb.Image(sample) for sample in samples],
                })

                # --- 重置训练损失累加器 ---
                # 为下一个 log_rate 周期重新从 0 开始累加
                acc_train_loss = 0

            # -------------------------------------------------------------------------
            # 步骤 6g: 定期保存 — 每 checkpoint_rate 步保存模型检查点
            # -------------------------------------------------------------------------
            # 检查当前迭代是否是检查点保存周期的倍数
            # checkpoint_rate 默认为 1000，与 log_rate 同步
            if iteration % args.checkpoint_rate == 0:
                # 构建模型检查点文件名
                # 命名格式：{项目名}-{运行名}-iteration-{迭代次数}-model.pth
                # 示例：ddpm-cifar10-ddpm-2026-07-17-09-30-iteration-1000-model.pth
                # 这种命名方式确保每个检查点有唯一的文件名，便于追溯
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"

                # 构建优化器检查点文件名（与模型检查点配套）
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                if not os.path.exists(args.log_dir):
                    os.mkdir(args.log_dir)

                # 保存模型权重到磁盘
                # diffusion.state_dict() 返回一个 OrderedDict，包含所有参数的名称和值
                # 注意：这里只保存模型权重，不保存模型架构，加载时需要重新构建模型
                torch.save(diffusion.state_dict(), model_filename)

                # 保存优化器状态到磁盘
                # optimizer.state_dict() 包含：
                #   - 'state': 每个参数的动量和二阶矩（Adam 的内部状态）
                #   - 'param_groups': 参数组的配置（学习率、betas 等）
                # 保存优化器状态确保从中断处恢复时，优化器的"记忆"不会丢失
                torch.save(optimizer.state_dict(), optim_filename)

        # -------------------------------------------------------------------------
        # 训练循环结束 — 正常完成
        # -------------------------------------------------------------------------
        # 当 iteration 达到 args.iterations 时，循环正常结束
        # 关闭 WandB 运行，将最终状态和所有缓冲的数据刷新到服务器
        if args.log_to_wandb:
            run.finish()

    # -------------------------------------------------------------------------
    # 异常处理 — 捕获 KeyboardInterrupt（用户按 Ctrl+C 中断）
    # -------------------------------------------------------------------------
    except KeyboardInterrupt:
        # 如果启用了 WandB，先完成运行（避免数据丢失）
        if args.log_to_wandb:
            run.finish()
        # 打印提示信息，告知用户训练被手动中断
        # 此时已保存的检查点可用于恢复训练
        print("Keyboard interrupt, run finished early")


# ==============================================================================
# 命令行参数解析器 — 配置所有训练超参数
# ==============================================================================
def create_argparser():
    """
    创建并配置 argparse.ArgumentParser 对象，定义所有可用的命令行参数及其默认值。

    参数来源：
      - 硬编码的 defaults 字典：训练脚本特有的参数
      - script_utils.diffusion_defaults()：扩散模型架构相关的参数

    合并后通过 script_utils.add_dict_to_argparser() 将所有键值对注册为 argparse 参数，
    最终返回配置好的 Parser 对象。
    """

    # -------------------------------------------------------------------------
    # 自动检测计算设备
    # -------------------------------------------------------------------------
    # torch.cuda.is_available() 检查是否有可用的 NVIDIA GPU 和正确安装的 CUDA
    # 如果有，使用 "cuda" 设备（GPU）；否则回退到 "cpu"
    # GPU 训练通常比 CPU 快 10-100 倍（取决于 GPU 型号）
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    # -------------------------------------------------------------------------
    # 生成运行名称（Run Name）
    # -------------------------------------------------------------------------
    # 使用当前日期时间生成唯一的运行名称，格式：ddpm-YYYY-MM-DD-HH-MM
    # 示例：ddpm-2026-07-17-09-53
    # 用途：在 WandB 上区分不同时间的训练运行，便于比较和管理
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")

    # -------------------------------------------------------------------------
    # 定义默认参数（训练脚本特有）
    # -------------------------------------------------------------------------
    defaults = dict(
        # ===== 优化器相关参数 =====
        learning_rate=2e-4,          # Adam 学习率：2e-4 是 DDPM 原始论文推荐值
                                     # 太大的学习率可能导致训练不稳定，太小的学习率收敛太慢

        batch_size=128,              # 训练批量大小：128 是 DDPM 默认值
                                     # 更大的 batch 提供更稳定的梯度估计，但需要更多 GPU 显存
                                     # CIFAR-10 图像 (3×32×32) × 128 ≈ 60MB 显存（仅图像数据）

        iterations=800000,           # 总训练迭代次数：800,000 步
                                     # 对于 CIFAR-10 (50,000 张图, batch=128)，约 2051 个 epoch
                                     # DDPM 原始论文在 CIFAR-10 上训练了 800,000 步

        # ===== WandB 日志相关参数 =====
        log_to_wandb=True,           # 是否启用 WandB 日志
                                     # True: 记录训练指标、样本图像到 WandB 网站
                                     # False: 关闭日志（节省网络带宽，适合本地调试）

        log_rate=1000,               # 日志记录频率：每 1000 步记录一次
                                     # 频率太高会增加网络开销，太低会丢失训练过程的细节

        checkpoint_rate=1000,        # 检查点保存频率：每 1000 步保存一次
                                     # 与 log_rate 同步，确保每次评估后都有对应的检查点

        log_dir=r"D:\GitHub\fmi_ddpm\ddpm_logs",       # 检查点保存目录：~ 代表用户主目录
                                     # 实际路径会被 shell 展开为 /home/user/ddpm_logs

        project_name='DDPM_CIFAR_TEST',           # WandB 项目名称：None 表示需要用户手动指定
                                     # 项目名用于在 WandB 网站上组织和分组实验
                                     # 示例："ddpm-cifar10"、"ddpm-cifar10-cosine" 等

        run_name=run_name,           # 运行名称：使用上面生成的时间戳名称
                                     # 每次训练有唯一的名称，便于在 WandB 上区分

        # ===== 检查点恢复相关参数 =====
        model_checkpoint=None,       # 预训练模型权重文件路径
                                     # None 表示从头开始训练
                                     # 如果指定路径，将加载该权重继续训练

        optim_checkpoint=None,       # 预训练优化器状态文件路径
                                     # None 表示从头开始（使用默认 Adam 初始状态）
                                     # 如果指定路径，将恢复 Adam 的动量和二阶矩状态

        # ===== 线性噪声调度相关参数 =====
        schedule_low=1e-4,           # 线性调度的 β 起始值：0.0001
                                     # DDPM 原始论文使用 β_1 = 1e-4

        schedule_high=0.02,          # 线性调度的 β 结束值：0.02
                                     # DDPM 原始论文使用 β_T = 0.02
                                     # 线性调度从低到高线性增加 β，意味着早期加噪慢、后期加噪快

        # ===== 设备相关参数 =====
        device=device,               # 计算设备：自动检测（GPU 优先，无 GPU 则用 CPU）
    )

    # -------------------------------------------------------------------------
    # 合并扩散模型架构的默认参数
    # -------------------------------------------------------------------------
    # defaults.update() 将 diffusion_defaults() 返回的字典合并到 defaults 中
    # diffusion_defaults() 定义在 script_utils.py 中，包含：
    #   - num_timesteps=1000          扩散总步数 T
    #   - schedule="linear"           噪声调度类型
    #   - loss_type="l2"              损失函数类型
    #   - use_labels=False            是否使用类别条件
    #   - base_channels=128           UNet 基础通道数
    #   - channel_mults=(1, 2, 2, 2)  UNet 通道倍率
    #   - num_res_blocks=2            每个 stage 残差块数
    #   - time_emb_dim=512            时间嵌入维度
    #   - norm="gn"                   归一化类型
    #   - dropout=0.1                 Dropout 率
    #   - activation="silu"           激活函数
    #   - attention_resolutions=(1,)  应用注意力的分辨率
    #   - ema_decay=0.9999            EMA 衰减率
    #   - ema_update_rate=1           EMA 更新频率
    defaults.update(script_utils.diffusion_defaults())

    # -------------------------------------------------------------------------
    # 创建 argparse 解析器并注册所有参数
    # -------------------------------------------------------------------------
    # argparse.ArgumentParser() 创建一个新的命令行参数解析器
    parser = argparse.ArgumentParser()

    # script_utils.add_dict_to_argparser() 将 defaults 字典中的每个键值对注册为 argparse 参数
    # 例如：learning_rate=2e-4 会被注册为 --learning_rate 2e-4
    # 类型自动推断：int、float、bool（通过 str2bool 转换）、str
    script_utils.add_dict_to_argparser(parser, defaults)

    return parser


# ==============================================================================
# 脚本入口点
# ==============================================================================
# __name__ == "__main__" 条件确保只有在直接运行此脚本时（而非被 import 时）才执行 main()
# 这是 Python 的标准做法，使文件既可以作为独立脚本运行，也可以作为模块被导入
if __name__ == "__main__":
    main()
