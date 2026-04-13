import os
import re
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def find_all_checkpoints(checkpoint_dir: str) -> List[str]:
    """查找指定目录下的所有checkpoint目录"""
    checkpoints = []
    checkpoint_dir = Path(checkpoint_dir)
    
    # 查找所有checkpoint-*格式的目录
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            checkpoints.append(str(item))
    
    # 按checkpoint编号排序
    checkpoints.sort(key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
    return checkpoints

def evaluate_checkpoint(checkpoint_path: str, test_data_path: str) -> Optional[Tuple[float, float]]:
    """评估单个checkpoint，返回(out_idx, out_token)分数"""
    try:
        # 运行评估命令
        cmd = ["python", "eval.py", test_data_path, checkpoint_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 从输出中提取分数
        output = result.stdout
        
        # 查找out_idx和out_token分数
        idx_match = re.search(r'out_idx:\s*([\d.]+)', output)
        token_match = re.search(r'out_token:\s*([\d.]+)', output)
        
        if idx_match and token_match:
            idx_score = float(idx_match.group(1))
            token_score = float(token_match.group(1))
            return idx_score, token_score
        else:
            print(f"警告: 无法从输出中提取分数: {checkpoint_path}")
            print(f"输出前200字符: {output[:200]}...")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"评估失败 {checkpoint_path}: {e}")
        print(f"stderr: {e.stderr[:200]}...")
        return None
    except Exception as e:
        print(f"评估错误 {checkpoint_path}: {e}")
        return None

def find_best_checkpoint(checkpoint_dir: str, test_data_path: str) -> Dict:
    """评估所有checkpoint并找到最佳模型"""
    # 查找所有checkpoint
    checkpoints = find_all_checkpoints(checkpoint_dir)
    print(f"找到 {len(checkpoints)} 个checkpoint")
    
    if not checkpoints:
        print("未找到任何checkpoint")
        return {}
    
    # 存储所有评估结果
    results = []
    
    for i, checkpoint_path in enumerate(checkpoints, 1):
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"[{i}/{len(checkpoints)}] 评估 {checkpoint_name}...")
        
        scores = evaluate_checkpoint(checkpoint_path, test_data_path)
        
        if scores:
            idx_score, token_score = scores
            results.append({
                'checkpoint': checkpoint_path,
                'checkpoint_name': checkpoint_name,
                'out_idx': idx_score,
                'out_token': token_score,
                'avg_score': (idx_score + token_score) / 2
            })
            print(f"  结果: out_idx={idx_score:.2f}, out_token={token_score:.2f}")
        else:
            print(f"  跳过: 评估失败")
    
    if not results:
        print("所有评估都失败了")
        return {}
    
    # 按两个分数都最高找出最佳模型
    # 先找出out_idx最高的
    max_idx = max(results, key=lambda x: x['out_idx'])['out_idx']
    candidates_idx = [r for r in results if r['out_idx'] == max_idx]
    
    # 在out_idx最高的候选中，找出out_token最高的
    if len(candidates_idx) > 1:
        best_result = max(candidates_idx, key=lambda x: x['out_token'])
    else:
        best_result = candidates_idx[0]
    
    # 输出所有结果表格
    print("\n" + "="*80)
    print("所有模型评估结果:")
    print("-"*80)
    print(f"{'序号':<5} {'Checkpoint':<20} {'out_idx':<10} {'out_token':<10} {'平均分':<10}")
    print("-"*80)
    
    for i, r in enumerate(results, 1):
        is_best = "*" if r['checkpoint'] == best_result['checkpoint'] else ""
        print(f"{i:<5} {r['checkpoint_name']:<20} {r['out_idx']:<10.2f} {r['out_token']:<10.2f} {r['avg_score']:<10.2f}{is_best}")
    
    # 输出最佳模型
    print("\n" + "="*80)
    print("🎯 最佳模型:")
    print(f"  路径: {best_result['checkpoint']}")
    print(f"  out_idx: {best_result['out_idx']:.2f}")
    print(f"  out_token: {best_result['out_token']:.2f}")
    print(f"  平均分: {best_result['avg_score']:.2f}")
    print("="*80)
    
    return best_result

def save_results_to_file(best_result: Dict, all_results: List[Dict], output_file: str = "best_model_result.json"):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_model': best_result,
            'all_results': all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 结果已保存到: {output_file}")

if __name__ == "__main__":
    # 配置路径
    test_data_path = "/home/zmt/layout_training/hetian/dataset_mydoc/dataset_category/test.jsonl.gz"
    checkpoint_dir = "/home/zmt/layout_training/hetian/dataset_mydoc/checkpoint/v3_categoryv1.1_2026-04-10-21"
    
    print("开始自动化评估所有模型...")
    print(f"测试数据: {test_data_path}")
    print(f"模型目录: {checkpoint_dir}")
    print("="*80)
    
    # 检查eval.py是否存在
    if not os.path.exists("eval.py"):
        print("错误: 当前目录下没有找到 eval.py 文件！")
        print("请确保在当前目录运行此脚本，或将脚本放在包含 eval.py 的目录中。")
        exit(1)
    
    # 运行评估
    best_result = find_best_checkpoint(checkpoint_dir, test_data_path)
    
