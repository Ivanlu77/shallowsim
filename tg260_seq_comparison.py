#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TG260 GPU在不同序列长度下的Prefill性能对比脚本
"""

import shallowsim as sb
import pandas as pd
import itertools

def main():
    print("=== TG260 GPU 不同序列长度性能对比 ===\n")
    
    # 初始化参数
    args = sb.ModelArgs()
    gpu_blackwell = sb.get_gpu_info('./device/gpu_info.csv', print_console=False)
    
    # 测试参数
    seq_len_list = [4383, 8766, 17532, 35064, 70128]
    kv_cache_rate = 0.563
    tp = 4  # Tensor Parallel
    dp = 8  # Data Parallel
    
    print(f"测试配置:")
    print(f"  序列长度: {seq_len_list}")
    print(f"  KV缓存命中率: {kv_cache_rate}")
    print(f"  TP: {tp}, DP: {dp}")
    print("=" * 60)
    
    # 收集TG260结果
    tg260_results = {}
    tg260_gpu = gpu_blackwell['TG260']
    
    print("正在计算不同序列长度下的性能...")
    for seq_len in seq_len_list:
        print(f"  计算 seq_len={seq_len}...")
        
        # 调用prefill_time获取单个GPU的结果
        gpu_dict = {'TG260': tg260_gpu}
        detail, summary = sb.prefill_time(args, gpu_dict, seq_len, kv_cache_rate, tp, dp, print_console=False)
        
        tg260_results[seq_len] = {
            'detail': detail['TG260'].to_dict(),
            'summary': summary['TG260'].to_dict(),
            'total_time_ms': summary.loc['Sum', 'TG260'],
            'throughput_tokens_per_sec_per_device': (seq_len / tp) * (1000 / summary.loc['Sum', 'TG260'])
        }
    
    print("\n" + "=" * 60)
    print("=== 详细算子性能对比 (毫秒) ===")
    
    # 创建详细性能表格
    detail_df = pd.DataFrame()
    detail_df.index.name = 'GPU'
    
    # 算子列表
    operators = ['MLA', 'DenseMLP', 'TP_MLA', 'Shared Expert', 'Combine', 'Overlap1', 
                'Routed Expert', 'Dispatch', 'Overlap2']
    
    for op in operators:
        row_data = {}
        for seq_len in seq_len_list:
            row_data[f'seq_{seq_len}'] = tg260_results[seq_len]['detail'][op]
        detail_df = pd.concat([detail_df, pd.DataFrame([row_data], index=[op])])
    
    print(detail_df.round(3).to_markdown())
    
    print("\n" + "=" * 60)
    print("=== 性能汇总对比 ===")
    
    # 创建汇总表格
    summary_df = pd.DataFrame()
    summary_metrics = ['Compute', 'Comm', 'Sum']
    
    for metric in summary_metrics:
        row_data = {}
        for seq_len in seq_len_list:
            row_data[f'seq_{seq_len}'] = tg260_results[seq_len]['summary'][metric]
        summary_df = pd.concat([summary_df, pd.DataFrame([row_data], index=[metric])])
    
    print(summary_df.round(3).to_markdown())
    
    print("\n" + "=" * 60)
    print("=== 吞吐量对比 (tokens/sec/device) ===")
    
    # 创建吞吐量表格
    throughput_data = {}
    for seq_len in seq_len_list:
        throughput_data[f'seq_{seq_len}'] = tg260_results[seq_len]['throughput_tokens_per_sec_per_device']
    
    throughput_df = pd.DataFrame([throughput_data], index=['TG260'])
    print(throughput_df.round(1).to_markdown())
    
    print("\n" + "=" * 60)
    print("=== 性能缩放效率分析 ===")
    
    print("序列长度翻倍的性能缩放效率:")
    for i in range(1, len(seq_len_list)):
        current_seq = seq_len_list[i]
        prev_seq = seq_len_list[i-1]
        
        current_time = tg260_results[current_seq]['total_time_ms']
        prev_time = tg260_results[prev_seq]['total_time_ms']
        
        scaling_ratio = current_time / prev_time
        theoretical_ratio = current_seq / prev_seq
        efficiency = theoretical_ratio / scaling_ratio
        
        print(f"  {prev_seq:5d} -> {current_seq:5d}: 实际={scaling_ratio:.2f}x, 理论={theoretical_ratio:.2f}x, 效率={efficiency:.1%}")
    
    print("\n" + "=" * 60)
    print("=== 算子时间占比分析 ===")
    
    # 分析各算子在总时间中的占比
    print("各算子在总计算时间中的占比 (%):")
    for seq_len in seq_len_list:
        print(f"\n序列长度 {seq_len}:")
        total_time = tg260_results[seq_len]['total_time_ms']
        detail_data = tg260_results[seq_len]['detail']
        
        n_sparse_layers = args.n_layers - args.n_dense_layers
        
        # 计算各部分实际贡献的时间
        mla_total = args.n_dense_layers * detail_data['MLA']
        dense_mlp_total = args.n_dense_layers * detail_data['DenseMLP']
        tp_mla_total = n_sparse_layers * detail_data['TP_MLA']
        shared_total = n_sparse_layers * detail_data['Shared Expert']
        routed_total = n_sparse_layers * detail_data['Routed Expert']
        
        print(f"  Dense MLA:     {mla_total/total_time*100:5.1f}% ({mla_total:6.1f}ms)")
        print(f"  Dense MLP:     {dense_mlp_total/total_time*100:5.1f}% ({dense_mlp_total:6.1f}ms)")
        print(f"  TP MLA:        {tp_mla_total/total_time*100:5.1f}% ({tp_mla_total:6.1f}ms)")
        print(f"  Shared Expert: {shared_total/total_time*100:5.1f}% ({shared_total:6.1f}ms)")
        print(f"  Routed Expert: {routed_total/total_time*100:5.1f}% ({routed_total:6.1f}ms)")
    
    print(f"\n脚本执行完成！")

if __name__ == "__main__":
    main() 