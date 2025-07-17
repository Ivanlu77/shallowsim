import shallowsim as sb

# 初始化
args = sb.ModelArgs()
gpu_dict = sb.get_gpu_info('./device/gpu_info.csv', print_console=False)

# 指定配置
gpu_name = 'TG260X-128'
tp, dp = 1, 128
seq_len = 4383
kv_cache_rate = 0.563

print(f"=== {gpu_name} 详细算子分析 ===")
print(f"配置: TP={tp}, DP={dp}")
print(f"序列长度: {seq_len}")
print(f"KV缓存命中率: {kv_cache_rate}")
print(f"设备总数: {tp * dp} = {tp * dp}")
print()

gpu = gpu_dict[gpu_name]

print("GPU基础信息:")
print(f"  SM数量: {gpu.sm}")
print(f"  通信SM: {gpu.comm_sm}")
print(f"  每节点GPU数: {gpu.gpu_per_node}")
print(f"  内存容量: {gpu.mem}GB")
print(f"  内存带宽: {gpu.mem_bw}GB/s")
print(f"  AR带宽: {gpu.ar_bw}GB/s") 
print(f"  A2A带宽: {gpu.a2a_bw}GB/s")
print(f"  PCIe带宽: {gpu.pcie_bw}GB/s")
print(f"  FP16算力: {gpu.fp16_flops}TFLOPS")
print(f"  FP8算力: {gpu.fp8_flops}TFLOPS")
print()

print("逐个算子详细分析:")
print("="*60)

# 1. Dense MLA
print("1. Dense MLA:")
dense_mla, tp_mla_dict = sb.mla_elapse_time(
    args, gpu, seq_len, kv_cache_rate,
    tp=[tp], decoding_mode=False, enable_gemm_fp4=True
)
print(f"  时间: {dense_mla:.6f}ms")
print(f"  TP={tp}时间: {tp_mla_dict[tp]:.6f}ms")
print()

# 2. Dense MLP  
print("2. Dense MLP:")
dense_mlp = sb._prefill_dense_mlp(args, gpu, seq_len)
print(f"  时间: {dense_mlp:.6f}ms")
print()

# 3. MOE计算
print("3. MOE Expert:")
shared, routed = sb._prefill_moe(args, gpu, seq_len, tp, dp)
print(f"  Shared Expert: {shared:.6f}ms")
print(f"  Routed Expert: {routed:.6f}ms")
print()

# 4. AllToAll通信
print("4. AllToAll通信:")
dispatch, combine = sb._prefill_alltoall(args, gpu, seq_len, tp, dp)
print(f"  Dispatch: {dispatch:.6f}ms")
print(f"  Combine: {combine:.6f}ms")

# 判断使用哪种带宽
device_number = tp * dp
if gpu.gpu_per_node < device_number:
    bandwidth_type = "PCIe"
    comm_bw = gpu.get_pcie_bw() * gpu.gpu_per_node
else:
    bandwidth_type = "A2A"
    comm_bw = gpu.get_a2a_bw()
print(f"  使用带宽类型: {bandwidth_type}")
print(f"  实际带宽: {comm_bw:.1f}GB/s")
print()

print("时间汇总:")
print("="*60)
print(f"Dense MLA:      {dense_mla:.6f}ms")
print(f"Dense MLP:      {dense_mlp:.6f}ms") 
print(f"TP MLA:         {tp_mla_dict[tp]:.6f}ms")
print(f"Shared Expert:  {shared:.6f}ms")
print(f"Routed Expert:  {routed:.6f}ms")
print(f"Combine:        {combine:.6f}ms")
print(f"Dispatch:       {dispatch:.6f}ms")
print()

# 计算overlap
overlap1 = combine - (tp_mla_dict[tp] + shared)
overlap2 = dispatch - routed

print("Overlap分析:")
print("="*60)
print(f"Overlap1 = Combine - (TP_MLA + Shared)")
print(f"        = {combine:.6f} - ({tp_mla_dict[tp]:.6f} + {shared:.6f})")
print(f"        = {overlap1:.6f}ms")
print()
print(f"Overlap2 = Dispatch - Routed")
print(f"        = {dispatch:.6f} - {routed:.6f}")
print(f"        = {overlap2:.6f}ms")
print()

# 最终时间计算
n_dense_layers = args.n_dense_layers
n_sparse_layers = args.n_layers - args.n_dense_layers

comp_time = n_dense_layers * (dense_mla + dense_mlp) + \
           n_sparse_layers * (tp_mla_dict[tp] + shared + routed)

sum_time = comp_time
if overlap1 > 0:
    sum_time += overlap1 * n_sparse_layers
if overlap2 > 0:
    sum_time += overlap2 * n_sparse_layers

print("最终时间计算:")
print("="*60)
print(f"Dense层数: {n_dense_layers}")
print(f"Sparse层数: {n_sparse_layers}")
print()
print(f"Dense部分时间: {n_dense_layers} × ({dense_mla:.6f} + {dense_mlp:.6f})")
print(f"              = {n_dense_layers * (dense_mla + dense_mlp):.6f}ms")
print()
print(f"Sparse部分时间: {n_sparse_layers} × ({tp_mla_dict[tp]:.6f} + {shared:.6f} + {routed:.6f})")
print(f"               = {n_sparse_layers * (tp_mla_dict[tp] + shared + routed):.6f}ms")
print()
print(f"总计算时间: {comp_time:.6f}ms")
print()

if overlap1 > 0:
    print(f"加上Overlap1: +{overlap1:.6f} × {n_sparse_layers} = +{overlap1 * n_sparse_layers:.6f}ms")
else:
    print(f"Overlap1 ≤ 0，通信被计算掩盖")

if overlap2 > 0:
    print(f"加上Overlap2: +{overlap2:.6f} × {n_sparse_layers} = +{overlap2 * n_sparse_layers:.6f}ms")
else:
    print(f"Overlap2 ≤ 0，通信被计算掩盖")

print()
print(f"最终总时间: {sum_time:.6f}ms = {sum_time/1000:.6f}s")
print(f"吞吐量: {seq_len / (sum_time/1000) / (tp * dp):.1f} tokens/sec/device") 