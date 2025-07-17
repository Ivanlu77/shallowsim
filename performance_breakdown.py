import shallowsim as sb

# 初始化
args = sb.ModelArgs()
gpu_dict = sb.get_gpu_info('./device/gpu_info.csv', print_console=False)

# 测试配置
seq_len = 4383
kv_cache_rate = 0.563
tp, dp = 1, 8  # 使用A2A带宽的配置

print("=== TG260系列性能分解分析 ===")
print(f"测试配置: seq_len={seq_len}, tp={tp}, dp={dp}, kv_cache_rate={kv_cache_rate}")
print()

tg260_gpus = ['TG260', 'TG260X-32', 'TG260X-64', 'TG260X-128', 'TG260X-288']

print(f"{'组件':15} {'TG260':10} {'TG260X-32':12} {'TG260X-64':12} {'TG260X-128':13} {'TG260X-288':13}")
print("=" * 95)

# 存储各组件时间
components = {}

for gpu_name in tg260_gpus:
    if gpu_name in gpu_dict:
        gpu = gpu_dict[gpu_name]
        
        # 计算各组件时间
        dense_mla, dense_mlp, tp_mla, shared, combine, routed, dispatch = sb._prefill_time(
            args, gpu, seq_len, kv_cache_rate, tp, dp
        )
        
        components[gpu_name] = {
            'dense_mla': dense_mla,
            'dense_mlp': dense_mlp,
            'tp_mla': tp_mla,
            'shared': shared,
            'combine': combine,
            'routed': routed,
            'dispatch': dispatch
        }

# 显示各组件时间
component_names = ['dense_mla', 'dense_mlp', 'tp_mla', 'shared', 'combine', 'routed', 'dispatch']
component_labels = ['Dense MLA', 'Dense MLP', 'TP MLA', 'Shared Expert', 'Combine', 'Routed Expert', 'Dispatch']

for comp_name, comp_label in zip(component_names, component_labels):
    row = f"{comp_label:15}"
    for gpu_name in tg260_gpus:
        if gpu_name in components:
            time_ms = components[gpu_name][comp_name] * 1000  # 转换为ms
            row += f"{time_ms:10.3f}"
        else:
            row += f"{'N/A':10}"
    print(row)

print()
print("=" * 95)

# 计算总时间和吞吐量
print(f"{'指标':15} {'TG260':10} {'TG260X-32':12} {'TG260X-64':12} {'TG260X-128':13} {'TG260X-288':13}")
print("=" * 95)

for gpu_name in tg260_gpus:
    if gpu_name in components:
        comp = components[gpu_name]
        
        # 按照prefill_time的逻辑计算总时间
        n_dense_layers = args.n_dense_layers
        n_sparse_layers = args.n_layers - args.n_dense_layers
        
        overlap1 = comp['combine'] - (comp['tp_mla'] + comp['shared'])
        overlap2 = comp['dispatch'] - comp['routed']
        
        comp_time = n_dense_layers * (comp['dense_mla'] + comp['dense_mlp']) + \
                   n_sparse_layers * (comp['tp_mla'] + comp['shared'] + comp['routed'])
        comm_time = n_sparse_layers * (comp['combine'] + comp['dispatch'])
        sum_time = comp_time
        if overlap1 > 0:
            sum_time += overlap1 * n_sparse_layers
        if overlap2 > 0:
            sum_time += overlap2 * n_sparse_layers
        
        # 计算吞吐量 (tokens/sec/device)
        throughput = seq_len / sum_time / (tp * dp)
        
        components[gpu_name]['total_time'] = sum_time
        components[gpu_name]['throughput'] = throughput

# 显示总时间和吞吐量
print(f"{'总时间(s)':15}", end="")
for gpu_name in tg260_gpus:
    if gpu_name in components:
        print(f"{components[gpu_name]['total_time']:10.6f}", end="")
    else:
        print(f"{'N/A':10}", end="")
print()

print(f"{'吞吐量(t/s/d)':15}", end="")
for gpu_name in tg260_gpus:
    if gpu_name in components:
        print(f"{components[gpu_name]['throughput']:10.1f}", end="")
    else:
        print(f"{'N/A':10}", end="")
print()

# 分析性能瓶颈
print("\n=== 性能瓶颈分析 ===")
for gpu_name in tg260_gpus:
    if gpu_name in components:
        comp = components[gpu_name]
        
        print(f"\n{gpu_name}:")
        print(f"  通信时间 (Combine + Dispatch): {(comp['combine'] + comp['dispatch'])*1000:.3f}ms")
        print(f"  计算时间 (其他组件): {(comp['dense_mla'] + comp['dense_mlp'] + comp['tp_mla'] + comp['shared'] + comp['routed'])*1000:.3f}ms")
        print(f"  通信时间占比: {((comp['combine'] + comp['dispatch']) / comp['total_time'])*100:.1f}%") 