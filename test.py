import torch
import time
import torch.nn.functional as F

def measure_tflops_conv(batch_size=32, in_channels=3, out_channels=64, kernel_size=3, input_size=224, num_iters=10, device='cuda'):
    """
    计算 TFLOPS 通过执行卷积运算，并测量执行时间。
    :param batch_size: 批次大小
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: 卷积核大小
    :param input_size: 输入图像大小 (H x W)
    :param num_iters: 迭代次数 (提高精度)
    :param device: 计算设备 ('cuda' 或 'cpu')
    :return: 计算出的 TFLOPS
    """
    if not torch.cuda.is_available() and device == 'cuda':
        raise RuntimeError("CUDA 设备不可用")
    
    torch.backends.cudnn.benchmark = True  # 让 cuDNN 选择最快的卷积算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32（仅在 Ampere 及以上架构上生效）
    torch.backends.cudnn.allow_tf32 = True  # 让 cuDNN 使用 TF32 加速
    
    dtype = torch.float16  # 使用 FP16 以触发 Tensor Core
    input_tensor = torch.randn(batch_size, in_channels, input_size, input_size, device=device, dtype=dtype)
    kernel = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device, dtype=dtype)
    
    # 预热 GPU (防止第一次运行时间异常)
    for _ in range(3):
        _ = F.conv2d(input_tensor, kernel, stride=1, padding=1)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    # 进行卷积运算
    for _ in range(num_iters):
        _ = F.conv2d(input_tensor, kernel, stride=1, padding=1)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / num_iters  # 平均时间
    output_size = input_size  # 假设 stride=1, padding=1 使得输出尺寸不变
    num_operations = batch_size * output_size * output_size * in_channels * out_channels * (kernel_size ** 2) * 2  # FLOP 计算
    tflops = (num_operations / elapsed_time) / 1e12  # 转换为 TFLOPS
    
    return tflops

# 运行测试（适用于 GPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tflops_result = measure_tflops_conv(batch_size=32, in_channels=3, out_channels=64, kernel_size=3, input_size=224, num_iters=10, device=device)
print(f"Estimated TFLOPS for FP16 convolution on {device}: {tflops_result:.2f} TFlops")