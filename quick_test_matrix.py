#!/usr/bin/env python3
"""
快速测试版本 - 生成完整的性能矩阵表格
"""

import shallowsim as sb
import pandas as pd
import itertools
import numpy as np

def main():
    print("=" * 80)
    print("ShallowSim Prefill 完整性能矩阵 - 快速测试")
    print("=" * 80)
    
    # 1. 初始化参数
    args = sb.ModelArgs()
    gpu_dict = sb.get_gpu_info('./device/gpu_info.csv', print_console=False)
    
    # 测试参数配置（简化版本用于快速测试）
    seq_len_list = [4383, 8766]  # 只测试2个序列长度
    kv_cache_rate = 0.563
    tp_list = [1, 2]             # 只测试2个TP值
    dp_list = [8, 16]            # 只测试2个DP值
    
    print(f"GPU类型({len(gpu_dict)}): {list(gpu_dict.keys())}")
    print(f"序列长度({len(seq_len_list)}): {seq_len_list}")
    print(f"TP配置({len(tp_list)}): {tp_list}")
    print(f"DP配置({len(dp_list)}): {dp_list}")
    print(f"总组合数: {len(seq_len_list) * len(tp_list) * len(dp_list)} = {len(seq_len_list)}×{len(tp_list)}×{len(dp_list)}")
    print("=" * 80)
    
    # 2. 批量测试并收集结果
    results = []
    total_combinations = len(seq_len_list) * len(tp_list) * len(dp_list)
    current_count = 0
    
    for seq_len, tp, dp in itertools.product(seq_len_list, tp_list, dp_list):
        current_count += 1
        combo_name = f"Seq{seq_len}_TP{tp}×DP{dp}"
        print(f"[{current_count:2d}/{total_combinations}] {combo_name}")
        
        try:
            # 计算prefill时间
            detail, summary = sb.prefill_time(args, gpu_dict, seq_len, kv_cache_rate, tp=tp, dp=dp, print_console=False)
            
            # 计算吞吐量 (tokens/second/device)
            throughput_per_device = summary.apply(lambda x: seq_len/tp * (1000 / x)).loc['Sum']
            
            # 保存每个GPU的结果
            for gpu_name in gpu_dict.keys():
                if gpu_name in summary.columns:
                    results.append({
                        'GPU': gpu_name,
                        'seq_len': seq_len,
                        'tp': tp,
                        'dp': dp,
                        'combo_name': combo_name,
                        'device_count': tp * dp,
                        'throughput': throughput_per_device[gpu_name],
                        'prefill_time_ms': summary.loc['Sum', gpu_name]
                    })
        
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            continue
    
    # 3. 转换为DataFrame
    results_df = pd.DataFrame(results)
    print(f"\n✅ 完成测试，共收集 {len(results_df)} 条结果")
    
    # 4. 生成完整的性能矩阵
    print("\n" + "=" * 120)
    print("📊 完整性能矩阵 (tokens/sec/device)")
    print("=" * 120)
    
    # 创建完整的透视表
    performance_matrix = results_df.pivot_table(
        index='GPU',
        columns='combo_name',
        values='throughput',
        aggfunc='first'
    ).round(1)
    
    # 按组合名称排序列（先按序列长度，再按TP，最后按DP）
    def sort_key(combo_name):
        # 解析 "Seq4383_TP1×DP8" 格式
        parts = combo_name.split('_')
        seq_len = int(parts[0][3:])  # 去掉 'Seq'
        tp_dp = parts[1]  # 'TP1×DP8'
        tp = int(tp_dp.split('×')[0][2:])  # 去掉 'TP'
        dp = int(tp_dp.split('×')[1][2:])  # 去掉 'DP'
        return (seq_len, tp, dp)
    
    sorted_columns = sorted(performance_matrix.columns, key=sort_key)
    performance_matrix = performance_matrix[sorted_columns]
    
    # 显示完整表格
    print(performance_matrix.to_string())
    
    # 5. 显示列的含义
    print(f"\n📋 列名含义:")
    for col in sorted_columns:
        parts = col.split('_')
        seq_len = parts[0][3:]
        tp_dp = parts[1]
        tp = tp_dp.split('×')[0][2:]
        dp = tp_dp.split('×')[1][2:]
        devices = int(tp) * int(dp)
        print(f"  {col}: 序列长度{seq_len}, TP{tp}, DP{dp}, 总设备数{devices}")
    
    # 6. 按序列长度分组显示
    print(f"\n" + "=" * 120)
    print("📊 按序列长度分组显示")
    print("=" * 120)
    
    for seq_len in seq_len_list:
        print(f"\n🎯 序列长度: {seq_len}")
        print("-" * 80)
        
        # 筛选当前序列长度的列
        seq_columns = [col for col in sorted_columns if col.startswith(f"Seq{seq_len}_")]
        seq_matrix = performance_matrix[seq_columns]
        
        # 重命名列，去掉序列长度前缀
        renamed_columns = {}
        for col in seq_columns:
            new_name = col.split('_')[1]  # 从 "Seq4383_TP1×DP8" 获取 "TP1×DP8"
            renamed_columns[col] = new_name
        
        seq_matrix_renamed = seq_matrix.rename(columns=renamed_columns)
        print(seq_matrix_renamed.to_string())
    
    # 7. 保存结果
    performance_matrix.to_csv("quick_test_matrix.csv")
    results_df.to_csv("quick_test_results.csv", index=False)
    
    print(f"\n💾 性能矩阵已保存到: quick_test_matrix.csv")
    print(f"💾 原始数据已保存到: quick_test_results.csv")
    
    print("\n" + "=" * 120)
    print("🎉 快速测试完成！")
    print("=" * 120)
    
    return results_df, performance_matrix

if __name__ == "__main__":
    results_df, performance_matrix = main() 