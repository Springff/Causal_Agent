"""
CausalAgent 生产级使用示例
展示完整的LLM工具调用能力和多智能体协作
"""

import logging
import json
import os
from datetime import datetime
from system_production import CausalAgentProduction

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_local_causal_inference():
    """
    示例1: 完整的局部因果推断工作流
    """
    print("\n" + "="*70)
    print("示例1: 局部因果推断工作流 (LLM-Powered)")
    print("="*70)
    
    system = CausalAgentProduction(verbose=True)

    
    research_goal = """
    识别与目标变量（致死率）存在潜在因果关系的基因？数据存储在："D:\Desktop\Causal_Agent\data\data.csv"，行是样本，列是基因，第一列是标签。
    """
    
    result = system.run_analysis(
        research_goal=research_goal.strip(),
        parameters={
            "target_variable": "致死率",
            "max_iterations": 5,
            "significance_level": 0.05
        }
    )
    
    print("\n[结果] 局部因果推断分析完成:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 保存结果
    output_path = system.save_results(output_dir="CAUSAL_AGENT/results/deg_analysis")
    print(f"✓ 详细结果已保存到: {output_path}")
    print(f"✓ 任务ID: {result.get('task_id')}")
    
    return result

def main():
    """
    运行所有生产级示例
    """
    print("\n" + "="*70)
    print("CausalAgent 生产级系统 - 完整工具调用演示")
    print("="*70)
    
    
    example_local_causal_inference()
    


if __name__ == "__main__":
    main()
