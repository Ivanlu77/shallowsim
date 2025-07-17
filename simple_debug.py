import shallowsim as sb

# 初始化
args = sb.ModelArgs()
gpu_dict = sb.get_gpu_info('./device/gpu_info.csv', print_console=False)

# 测试TG260系列
tg260_gpus = ['TG260', 'TG260X-32', 'TG260X-64']

print("=== TG260系列A2A带宽测试 ===")
print(f"{'GPU':12} {'gpu_per_node':12} {'a2a_bw':8} {'pcie_bw':8}")
print("-" * 50)

for gpu_name in tg260_gpus:
    if gpu_name in gpu_dict:
        gpu = gpu_dict[gpu_name]
        print(f"{gpu_name:12} {gpu.gpu_per_node:12} {gpu.a2a_bw:8} {gpu.pcie_bw:8}")

print("\n=== 测试配置 TP=1, DP=8 ===")
tp, dp = 1, 8
device_number = tp * dp

for gpu_name in tg260_gpus:
    if gpu_name in gpu_dict:
        gpu = gpu_dict[gpu_name]
        
        # 判断使用哪种带宽
        if gpu.gpu_per_node < device_number:
            bandwidth_type = "PCIe"
            comm_bw = gpu.get_pcie_bw() * gpu.gpu_per_node
        else:
            bandwidth_type = "A2A"
            comm_bw = gpu.get_a2a_bw()
        
        # 计算通信时间
        dispatch_time, combine_time = sb._prefill_alltoall(args, gpu, 4383, tp, dp)
        
        print(f"{gpu_name:12}: 设备数={device_number:2d}, gpu_per_node={gpu.gpu_per_node:3d} → {bandwidth_type:4s} BW={comm_bw:6.1f} → Dispatch={dispatch_time:.6f}ms")

print("\n=== 测试配置 TP=1, DP=16 ===")
tp, dp = 1, 16
device_number = tp * dp

for gpu_name in tg260_gpus:
    if gpu_name in gpu_dict:
        gpu = gpu_dict[gpu_name]
        
        if gpu.gpu_per_node < device_number:
            bandwidth_type = "PCIe"
            comm_bw = gpu.get_pcie_bw() * gpu.gpu_per_node
        else:
            bandwidth_type = "A2A"
            comm_bw = gpu.get_a2a_bw()
        
        dispatch_time, combine_time = sb._prefill_alltoall(args, gpu, 4383, tp, dp)
        
        print(f"{gpu_name:12}: 设备数={device_number:2d}, gpu_per_node={gpu.gpu_per_node:3d} → {bandwidth_type:4s} BW={comm_bw:6.1f} → Dispatch={dispatch_time:.6f}ms") 