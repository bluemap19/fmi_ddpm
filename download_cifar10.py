import os
import torchvision
from torchvision import transforms

# --- 调试代码开始 ---
print("--- 开始调试路径 ---")

# 1. 打印当前工作目录 (非常重要！)
current_working_dir = os.getcwd()
print(f"当前工作目录 (CWD): {current_working_dir}")

# 2. 定义 data 目录的路径
data_dir = './data'
print(f"\n程序期望的 data 目录路径: {os.path.abspath(data_dir)}")

# 3. 检查 data 目录是否存在，并列出其内容
if os.path.exists(data_dir):
    print(f"\n'{data_dir}' 目录存在。")
    print("目录下的文件和文件夹:")
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        print(f"  - {item} (是文件夹: {os.path.isdir(item_path)})")

    # 4. 如果 cifar-10-batches-py 文件夹存在，也检查一下
    cifar_folder = os.path.join(data_dir, 'cifar-10-batches-py')
    if os.path.exists(cifar_folder):
        print(f"\n'{cifar_folder}' 文件夹存在。")
        print("文件夹下的文件:")
        for file in os.listdir(cifar_folder):
            print(f"  - {file}")
    else:
        os.makedirs(cifar_folder, exist_ok=True)
        # print(f"\n'{cifar_folder}' 文件夹不存在！这是问题的关键！")

else:
    os.makedirs(data_dir, exist_ok=True)
    print(f"\n'{data_dir}' 目录不存在！")

print("--- 调试路径结束 ---\n")
# --- 调试代码结束 ---

# --- 原有的加载代码 ---
print("开始加载 CIFAR-10 数据集...")

try:
    train_data = torchvision.datasets.CIFAR10(
        root='./data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    print("数据集加载成功!")
    print(f"训练集样本数: {len(train_data)}")

except RuntimeError as e:
    print(f"\n加载失败，报错信息: {e}")
    print("\n请根据上面的调试信息，确认文件路径是否正确。")