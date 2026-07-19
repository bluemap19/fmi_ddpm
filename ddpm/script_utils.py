import argparse
import torchvision
import torch.nn.functional as F

from ddpm.unet import  UNet
from ddpm.diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data

# def get_transform():
#     class RescaleChannels(object):
#         def __call__(self, sample):
#             return 2 * sample - 1
#
#     return torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         RescaleChannels(),
#     ])
class RescaleChannels:
    def __call__(self, pic):
        return pic * 2 - 1

def get_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


"""MNIST专用的数据变换，转换为3通道32x32"""
class To3Channels:
    def __call__(self, x):
        # x的形状是(1, 32, 32)，转换为(3, 32, 32)
        return x.expand(3, -1, -1)

# 在 script_utils.py 中添加
def get_transform_mnist():

    """MNIST专用的数据变换"""
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),  # MNIST是28x28，调整到32x32以匹配原模型
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 复制单通道为3通道
        # torchvision.transforms.Grayscale(num_output_channels=3),  # 直接转换为3通道
        To3Channels(),  # 使用自定义的转换类
        RescaleChannels(),  # 将[0,1]映射到[-1,1]
    ])

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=False,

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,
    )

    return defaults


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    model = UNet(
        img_channels=3,

        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,

        num_classes=None if not args.use_labels else 10,
        initial_pad=0,
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model, (32, 32), 3, 10,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
    )

    return diffusion