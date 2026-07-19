import argparse
import torch
import torchvision
from ddpm import script_utils


def main():
    args = create_argparser().parse_args()
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load(args.model_path))

        if args.use_labels:
            for label in range(10):
                y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
                samples = diffusion.sample(args.num_images // 10, device, y=y)

                for image_id in range(len(samples)):
                    image = ((samples[image_id] + 1) / 2).clip(0, 1)
                    torchvision.utils.save_image(image, f"{args.save_dir}/{label}-{image_id}.png")
        else:
            samples = diffusion.sample(args.num_images, device)

            for image_id in range(len(samples)):
                image = ((samples[image_id] + 1) / 2).clip(0, 1)
                torchvision.utils.save_image(image, f"{args.save_dir}/{image_id}.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(
        num_images=100,
        device=device,
        schedule_low=1e-4,  # 线性调度的 β 起始值：0.0001,DDPM 原始论文使用 β_1 = 1e-4
        schedule_high=0.02,  # 线性调度的 β 结束值：0.02
        # DDPM 原始论文使用 β_T = 0.02
        # 线性调度从低到高线性增加 β，意味着早期加噪慢、后期加噪快
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=r'D:\GitHub\fmi_ddpm\ddpm_logs\DDPM_CIFAR_TEST-ddpm-2026-07-18-21-22-iteration-68000-model.pth', type=str)
    parser.add_argument("--save_dir", default=r'D:\GitHub\fmi_ddpm\ddpm_testresult', type=str)
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()