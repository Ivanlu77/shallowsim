import shallowsim as sb

# 初始化
args = sb.ModelArgs()
gpu_dict = sb.get_gpu_info('./device/gpu_info.csv', print_console=False)

# 测试配置
seq_len = 4383
kv_cache_rate = 0.563
tp, dp = 1, 8

print("=== Overlap计算分析 ===")
print(f"n_dense_layers = {args.n_dense_layers}")
print(f"n_sparse_layers = {args.n_layers - args.n_dense_layers}")
print()

tg260_gpus = ['TG260', 'TG260X-32']  # 只看两个GPU对比

for gpu_name in tg260_gpus:
    if gpu_name in gpu_dict:
        gpu = gpu_dict[gpu_name]
        
        # 计算各组件时间
        dense_mla, dense_mlp, tp_mla, shared, combine, routed, dispatch = sb._prefill_time(
            args, gpu, seq_len, kv_cache_rate, tp, dp
        )
        
        print(f"=== {gpu_name} ===")
        print(f"dense_mla: {dense_mla:.3f}ms")
        print(f"dense_mlp: {dense_mlp:.3f}ms") 
        print(f"tp_mla: {tp_mla:.3f}ms")
        print(f"shared: {shared:.3f}ms")
        print(f"combine: {combine:.3f}ms")
        print(f"routed: {routed:.3f}ms")
        print(f"dispatch: {dispatch:.3f}ms")
        
        # 计算overlap
        overlap1 = combine - (tp_mla + shared)
        overlap2 = dispatch - routed
        
        print(f"\nOverlap计算:")
        print(f"overlap1 = combine - (tp_mla + shared) = {combine:.3f} - ({tp_mla:.3f} + {shared:.3f}) = {overlap1:.3f}ms")
        print(f"overlap2 = dispatch - routed = {dispatch:.3f} - {routed:.3f} = {overlap2:.3f}ms")
        
        # 计算各部分时间
        n_dense_layers = args.n_dense_layers
        n_sparse_layers = args.n_layers - args.n_dense_layers
        
        comp_time = n_dense_layers * (dense_mla + dense_mlp) + \
                   n_sparse_layers * (tp_mla + shared + routed)
        
        print(f"\n计算时间:")
        print(f"dense部分: {n_dense_layers} * ({dense_mla:.3f} + {dense_mlp:.3f}) = {n_dense_layers * (dense_mla + dense_mlp):.3f}ms")
        print(f"sparse部分: {n_sparse_layers} * ({tp_mla:.3f} + {shared:.3f} + {routed:.3f}) = {n_sparse_layers * (tp_mla + shared + routed):.3f}ms")
        print(f"总计算时间: {comp_time:.3f}ms")
        
        # 通信时间
        comm_time = n_sparse_layers * (combine + dispatch)
        print(f"\n通信时间:")
        print(f"sparse层通信: {n_sparse_layers} * ({combine:.3f} + {dispatch:.3f}) = {comm_time:.3f}ms")
        
        # 最终时间
        sum_time = comp_time
        if overlap1 > 0:
            sum_time += overlap1 * n_sparse_layers
            print(f"加上overlap1: +{overlap1:.3f} * {n_sparse_layers} = +{overlap1 * n_sparse_layers:.3f}ms")
        if overlap2 > 0:
            sum_time += overlap2 * n_sparse_layers
            print(f"加上overlap2: +{overlap2:.3f} * {n_sparse_layers} = +{overlap2 * n_sparse_layers:.3f}ms")
        
        print(f"最终总时间: {sum_time:.3f}ms")
        print(f"吞吐量: {seq_len / (sum_time/1000) / (tp * dp):.1f} tokens/sec/device")
        print() 