#!/usr/bin/env python3
"""
生成完整的性能矩阵表格
包含所有TP×DP组合与序列长度组合的不同GPU卡的结果
"""

import shallowsim as sb
import pandas as pd
import itertools
import numpy as np
from datetime import datetime

def main():
    print("=" * 80)
    print("ShallowSim Prefill 完整性能矩阵")
    print("=" * 80)
    
    # 1. 初始化参数
    args = sb.ModelArgs()
    gpu_dict = sb.get_gpu_info('./device/gpu_info.csv', print_console=False)
    
    # 测试参数配置
    seq_len_list = [4383, 8766, 17532, 35064, 70128]  # 序列长度列表
    kv_cache_rate = 0.563                              # KV缓存命中率
    tp_list = [1, 2, 4]                               # Tensor Parallel 配置
    dp_list = [8, 16, 32, 64, 128]                    # Data Parallel 配置
    
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
            # 即使出错也要记录，用NaN填充
            for gpu_name in gpu_dict.keys():
                results.append({
                    'GPU': gpu_name,
                    'seq_len': seq_len,
                    'tp': tp,
                    'dp': dp,
                    'combo_name': combo_name,
                    'device_count': tp * dp,
                    'throughput': np.nan,
                    'prefill_time_ms': np.nan
                })
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
    
    # 5. 生成分组显示的表格（按序列长度分组）
    print(f"\n" + "=" * 120)
    print("📊 按序列长度分组的性能表格")
    print("=" * 120)
    
    for seq_len in seq_len_list:
        print(f"\n🎯 序列长度: {seq_len}")
        print("-" * 100)
        
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
        
        # 显示设备总数信息
        print(f"\n设备总数对照:")
        device_info = []
        for tp in tp_list:
            for dp in dp_list:
                device_info.append(f"TP{tp}×DP{dp}={tp*dp}")
        print(" | ".join(device_info))
    
    # 6. 统计分析
    print(f"\n" + "=" * 120)
    print("📊 统计分析")
    print("=" * 120)
    
    # 6.1 各GPU的最佳配置
    print("\n🏆 各GPU的最佳性能配置:")
    best_configs = results_df.loc[results_df.groupby('GPU')['throughput'].idxmax()]
    best_summary = best_configs[['GPU', 'combo_name', 'device_count', 'throughput', 'prefill_time_ms']].round(1)
    best_summary = best_summary.sort_values('throughput', ascending=False)
    print(best_summary.to_string(index=False))
    
    # 6.2 各序列长度的最佳配置
    print("\n📈 各序列长度的最佳性能配置:")
    best_seq_configs = results_df.loc[results_df.groupby('seq_len')['throughput'].idxmax()]
    best_seq_summary = best_seq_configs[['seq_len', 'GPU', 'tp', 'dp', 'device_count', 'throughput']].round(1)
    best_seq_summary = best_seq_summary.sort_values('seq_len')
    print(best_seq_summary.to_string(index=False))
    
    # 6.3 各TP×DP组合的最佳性能
    print("\n⚙️  各TP×DP组合的最佳性能:")
    results_df['tp_dp_combo'] = results_df.apply(lambda x: f"TP{x['tp']}×DP{x['dp']}", axis=1)
    best_combo_configs = results_df.loc[results_df.groupby('tp_dp_combo')['throughput'].idxmax()]
    best_combo_summary = best_combo_configs[['tp_dp_combo', 'GPU', 'seq_len', 'device_count', 'throughput']].round(1)
    best_combo_summary = best_combo_summary.sort_values('throughput', ascending=False)
    print(best_combo_summary.to_string(index=False))
    
    # 7. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存原始数据
    csv_filename = f"prefill_complete_matrix_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    
    # 保存性能矩阵
    matrix_filename = f"prefill_performance_matrix_{timestamp}.csv"
    performance_matrix.to_csv(matrix_filename)
    
    print(f"\n💾 原始数据已保存到: {csv_filename}")
    print(f"💾 性能矩阵已保存到: {matrix_filename}")
    
    # 8. 生成Excel文件（多个工作表）
    excel_filename = f"prefill_complete_analysis_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 完整性能矩阵
        performance_matrix.to_excel(writer, sheet_name='完整性能矩阵')
        
        # 按序列长度分组的表格
        for seq_len in seq_len_list:
            seq_columns = [col for col in sorted_columns if col.startswith(f"Seq{seq_len}_")]
            seq_matrix = performance_matrix[seq_columns]
            renamed_columns = {col: col.split('_')[1] for col in seq_columns}
            seq_matrix_renamed = seq_matrix.rename(columns=renamed_columns)
            seq_matrix_renamed.to_excel(writer, sheet_name=f'Seq{seq_len}')
        
        # 统计汇总
        best_summary.to_excel(writer, sheet_name='最佳配置汇总', index=False)
        
        # 原始数据
        results_df.to_excel(writer, sheet_name='原始数据', index=False)
    
    print(f"📊 Excel分析报告已保存到: {excel_filename}")
    
    # 9. 生成Markdown报告
    markdown_filename = f"prefill_complete_report_{timestamp}.md"
    with open(markdown_filename, 'w', encoding='utf-8') as f:
        f.write("# ShallowSim Prefill 完整性能矩阵报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 测试配置\n\n")
        f.write(f"- GPU类型: {list(gpu_dict.keys())}\n")
        f.write(f"- 序列长度: {seq_len_list}\n")
        f.write(f"- TP配置: {tp_list}\n") 
        f.write(f"- DP配置: {dp_list}\n")
        f.write(f"- KV缓存命中率: {kv_cache_rate}\n")
        f.write(f"- 总测试组合: {len(seq_len_list) * len(tp_list) * len(dp_list)}\n\n")
        
        f.write("## 完整性能矩阵 (tokens/sec/device)\n\n")
        f.write(performance_matrix.to_markdown())
        
        f.write("\n\n## 最佳性能配置汇总\n\n")
        f.write(best_summary.to_markdown(index=False))
    
    print(f"📄 Markdown报告已保存到: {markdown_filename}")
    
    print("\n" + "=" * 120)
    print("🎉 完整矩阵分析完成！")
    print(f"📋 测试了 {len(gpu_dict)} 种GPU × {len(seq_len_list) * len(tp_list) * len(dp_list)} 种配置组合")
    print("=" * 120)
    
    return results_df, performance_matrix

if __name__ == "__main__":
    results_df, performance_matrix = main() 