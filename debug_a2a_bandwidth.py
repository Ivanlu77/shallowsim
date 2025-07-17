#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本：验证TG260系列GPU的A2A带宽使用情况
"""

import sys
import os
sys.path.insert(0, '.')

import shallowsim as sb
import pandas as pd

def debug_a2a_usage():
    """调试A2A带宽的使用情况"""
    
    # 初始化模型参数
    args = sb.ModelArgs()
    
    # 获取GPU信息
    gpu_dict = sb.get_gpu_info('./device/gpu_info.csv', print_console=False)
    
    # 测试配置
    tp_list = [1, 2, 4]
    dp_list = [8, 16, 32, 64, 128]
    seq_len = 4383
    
    print("=== TG260系列GPU A2A带宽使用分析 ===\n")
    
    # 只分析TG260系列
    tg260_gpus = {k: v for k, v in gpu_dict.items() if k.startswith('TG260')}
    
    results = []
    
    for gpu_name, gpu in tg260_gpus.items():
        print(f"\n🔍 {gpu_name} (gpu_per_node={gpu.gpu_per_node}, a2a_bw={gpu.a2a_bw}, pcie_bw={gpu.pcie_bw})")
        print("-" * 60)
        
        for tp in tp_list:
            for dp in dp_list:
                device_number = tp * dp
                
                # 复制_prefill_alltoall的逻辑来判断使用哪种带宽
                if gpu.gpu_per_node < device_number:
                    bandwidth_type = "PCIe"
                    comm_bw = gpu.get_pcie_bw() * gpu.gpu_per_node
                else:
                    bandwidth_type = "A2A"
                    comm_bw = gpu.get_a2a_bw()
                
                # 计算实际的dispatch和combine时间
                dispatch_time, combine_time = sb._prefill_alltoall(
                    args, gpu, seq_len, tp, dp
                )
                
                print(f"  TP={tp:1d}, DP={dp:3d} → 设备数={device_number:3d} → {bandwidth_type:4s} → BW={comm_bw:5.1f} → Dispatch={dispatch_time:.3f}ms")
                
                results.append({
                    'GPU': gpu_name,
                    'gpu_per_node': gpu.gpu_per_node,
                    'tp': tp,
                    'dp': dp,
                    'device_number': device_number,
                    'bandwidth_type': bandwidth_type,
                    'comm_bw': comm_bw,
                    'dispatch_time': dispatch_time,
                    'combine_time': combine_time,
                    'a2a_bw': gpu.a2a_bw,
                    'pcie_bw': gpu.pcie_bw
                })
    
    # 转换为DataFrame进行分析
    df = pd.DataFrame(results)
    
    print(f"\n\n=== 汇总分析 ===")
    print(f"总测试配置数: {len(df)}")
    print(f"使用A2A带宽的配置数: {len(df[df['bandwidth_type'] == 'A2A'])}")
    print(f"使用PCIe带宽的配置数: {len(df[df['bandwidth_type'] == 'PCIe'])}")
    
    # 显示使用A2A带宽的配置
    a2a_configs = df[df['bandwidth_type'] == 'A2A']
    if len(a2a_configs) > 0:
        print(f"\n📊 使用A2A带宽的配置:")
        for _, row in a2a_configs.iterrows():
            print(f"  {row['GPU']}: TP={row['tp']}, DP={row['dp']} → A2A_BW={row['a2a_bw']}")
    
    # 检查不同GPU在相同配置下的性能差异
    print(f"\n🔍 相同配置下的性能差异检查:")
    for tp in [1, 2]:
        for dp in [8, 16]:
            subset = df[(df['tp'] == tp) & (df['dp'] == dp)]
            if len(subset) > 1:
                dispatch_times = subset['dispatch_time'].values
                if len(set([round(t, 6) for t in dispatch_times])) > 1:
                    print(f"  TP={tp}, DP={dp}:")
                    for _, row in subset.iterrows():
                        print(f"    {row['GPU']}: {row['dispatch_time']:.6f}ms ({row['bandwidth_type']})")
                else:
                    print(f"  TP={tp}, DP={dp}: 所有GPU性能相同 ({dispatch_times[0]:.6f}ms)")
    
    return df

if __name__ == "__main__":
    debug_df = debug_a2a_usage()
    debug_df.to_csv('debug_a2a_bandwidth_analysis.csv', index=False)
    print(f"\n详细结果已保存到: debug_a2a_bandwidth_analysis.csv") 