"""
BioInfoMAS 生产级使用示例
展示完整的LLM工具调用能力和多智能体协作
"""

import logging
import json
import os
from datetime import datetime
from system_production import BioInfoMASProduction

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_differential_expression_analysis():
    """
    示例1: 完整的差异表达分析工作流 - 使用真实LLM调用
    包括数据获取、质量控制、分析和可视化
    """
    print("\n" + "="*70)
    print("示例1: 差异表达分析工作流 (LLM-Powered)")
    print("="*70)
    
    system = BioInfoMASProduction(verbose=True)

    #下载GEO数据库中GSE12456数据，进行差异表达分析，识别与阿尔茨海默病(AD)相关的基因和通路：
    
    research_goal = """
    分析AD数据（储存路径：D:\Desktop\Agent\BioInfoMAS\data\AD.h5ad），识别其中经常出现的细胞模体，并分析其生物学意义。
    """
    
    result = system.run_analysis(
        research_goal=research_goal.strip(),
        parameters={
            "fdr_threshold": 0.05,
            "log2fc_threshold": 1.0,
            "min_sample_size": 10,
            "database": "kegg",
            "output_format": "html",
            "include_literature": True
        }
    )
    
    print("\n[结果] 差异表达分析完成:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 保存结果
    output_path = system.save_results(output_dir="BioInfoMAS/results/deg_analysis")
    print(f"✓ 详细结果已保存到: {output_path}")
    print(f"✓ 任务ID: {result.get('task_id')}")
    
    return result


def example_variant_calling_analysis():
    """
    示例2: 全基因组变异检测分析 - 使用真实LLM调用
    包括序列比对、变异检测和注释
    """
    print("\n" + "="*70)
    print("示例2: 全基因组变异检测 (LLM-Powered)")
    print("="*70)
    
    system = BioInfoMASProduction(verbose=True)
    
    research_goal = """
    进行全基因组测序(WGS)分析，识别与神经退行性疾病相关的变异：
    
    研究设计：
    - 患者样本：帕金森病患者(n=100)
    - 对照组：健康志愿者(n=100)
    - 测序深度：30x whole genome coverage
    - 平台：Illumina NovaSeq 6000
    
    分析步骤：
    1. 数据准备：QC质控，下载与GRCh38参考基因组比对
    2. 变异检测：识别SNP、Indel和结构变异
    3. 变异注释：使用VEP预测功能影响
    4. 致病性评估：查询ClinVar、gnomAD等数据库
    5. 知识推理：搜索与帕金森病相关的变异文献
    
    输出要求：
    - 识别与疾病相关的致病变异
    - 评估变异的临床意义
    - 生成变异分布的可视化图表
    - 提供医学遗传学解释
    """
    
    result = system.run_analysis(
        research_goal=research_goal.strip(),
        parameters={
            "reference_genome": "GRCh38",
            "min_quality": 20,
            "variant_types": ["SNP", "INDEL", "SV"],
            "annotate": True,
            "clinical_significance": True,
            "include_clin_var": True
        }
    )
    
    print("\n[结果] 全基因组变异检测完成:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 保存结果
    output_path = system.save_results(output_dir="./results/wgs_analysis")
    print(f"✓ 详细结果已保存到: {output_path}")
    print(f"✓ 任务ID: {result.get('task_id')}")
    
    return result


def example_single_cell_analysis():
    """
    示例3: 单细胞RNA-seq分析 - 使用真实LLM调用
    包括细胞聚类、细胞类型注释和细胞间通信
    """
    print("\n" + "="*70)
    print("示例3: 单细胞RNA-seq分析 (LLM-Powered)")
    print("="*70)
    
    system = BioInfoMASProduction(verbose=True)
    
    research_goal = """
    分析单细胞RNA-seq数据，研究肿瘤微环境中的细胞异质性：
    
    实验设计：
    - 样本来源：乳腺癌患者肿瘤组织
    - 技术平台：10x Genomics (3' chemistry)
    - 细胞捕获：~15000个细胞
    - 测序深度：>50000 reads/cell
    
    分析流程：
    1. 原始数据处理：质控、过滤低质量细胞和基因
    2. 数据标准化：SCT/Log-normalize，批次效应校正(Harmony)
    3. 降维与聚类：PCA->UMAP，Leiden聚类(resolution=0.6)
    4. 细胞类型注释：使用标记基因和单细胞参考数据库
    5. 细胞间通信：CellChat分析关键细胞群体间的相互作用
    
    输出要求：
    - 鉴定肿瘤微环境中的主要细胞类型
    - 分析T细胞亚群和耗尽特征
    - 识别肿瘤相关成纤维细胞(CAF)的亚群
    - 绘制关键通路的细胞间通信网络
    """
    
    result = system.run_analysis(
        research_goal=research_goal.strip(),
        parameters={
            "min_genes": 200,
            "max_genes": 2500,
            "min_cells": 3,
            "n_clusters": 12,
            "resolution": 0.6,
            "batch_correction": True,
            "cellchat_analysis": True
        }
    )
    
    print("\n[结果] 单细胞RNA-seq分析完成:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 保存结果
    output_path = system.save_results(output_dir="./results/scrnaseq_analysis")
    print(f"✓ 详细结果已保存到: {output_path}")
    print(f"✓ 任务ID: {result.get('task_id')}")
    
    return result


def example_pathway_and_network_analysis():
    """
    示例4: 生物通路和网络分析 - 使用真实LLM调用
    包括通路富集、网络构建和模块检测
    """
    print("\n" + "="*70)
    print("示例4: 生物通路和网络分析 (LLM-Powered)")
    print("="*70)
    
    system = BioInfoMASProduction(verbose=True)
    
    research_goal = """
    进行深入的生物通路和网络分析，揭示疾病机制：
    
    分析目标：
    - 输入基因集：前期差异表达分析得到的2543个差异基因
    - 研究疾病：乳腺癌的分子特征和关键通路
    
    分析流程：
    1. 基因集准备：过滤和标准化基因列表
    2. 功能富集分析：
       - KEGG通路富集 (FDR < 0.05)
       - Gene Ontology注释 (BP, CC, MF)
       - Reactome pathway分析
    3. 蛋白质相互作用：
       - 构建PPI网络 (String数据库)
       - Louvain聚类检测关键模块
       - 中心性分析识别hub基因
    4. 知识图谱整合：
       - 查询DisGeNET疾病-基因关联
       - 整合TCGA肿瘤特征
    5. 可视化输出：
       - 通路相互作用网络图
       - 网络拓扑统计分析
       - 医学遗传学解释
    
    输出产品：
    - 前10个显著富集的KEGG通路
    - 关键PPI模块及其功能
    - 候选肿瘤相关基因排名
    """
    
    result = system.run_analysis(
        research_goal=research_goal.strip(),
        parameters={
            "databases": ["kegg", "go", "reactome", "disgenet"],
            "p_value_cutoff": 0.05,
            "fdr_cutoff": 0.05,
            "min_pathway_size": 3,
            "build_network": True,
            "network_database": "string",
            "clustering_method": "louvain"
        }
    )
    
    print("\n[结果] 通路和网络分析完成:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 保存结果
    output_path = system.save_results(output_dir="./results/pathway_analysis")
    print(f"✓ 详细结果已保存到: {output_path}")
    print(f"✓ 任务ID: {result.get('task_id')}")
    
    return result


def example_batch_analysis():
    """
    示例5: 批量分析多个样本 - 使用真实LLM调用
    展示系统的可扩展性和效率
    """
    print("\n" + "="*70)
    print("示例5: 批量分析模式 (LLM-Powered)")
    print("="*70)
    
    system = BioInfoMASProduction(verbose=False)
    
    # 定义多个真实的分析任务
    analysis_tasks = [
        {
            "name": "肺癌(LUAD)",
            "goal": """
                对肺腺癌TCGA数据集进行完整分析：
                - 对比肿瘤与正常组织的基因表达
                - 识别肺癌特征基因和通路
                - 预测患者预后和治疗响应
                - 生成临床决策支持信息
            """,
            "parameters": {
                "cancer_type": "LUAD",
                "dataset": "TCGA-LUAD",
                "min_samples": 30
            }
        },
        {
            "name": "结肠癌(COAD)",
            "goal": """
                对结肠癌进行分子分层分析：
                - 鉴定结肠癌分子亚型
                - 分析MSI与CMS分类
                - 识别关键驱动基因
                - 评估免疫浸润特征
            """,
            "parameters": {
                "cancer_type": "COAD",
                "dataset": "TCGA-COAD",
                "include_msi_analysis": True
            }
        },
        {
            "name": "前列腺癌(PRAD)",
            "goal": """
                进行前列腺癌多组学分析：
                - 整合基因表达和突变数据
                - 识别预后相关基因
                - 分析雄激素信号通路
                - 预测去势抵抗型前列腺癌风险
            """,
            "parameters": {
                "cancer_type": "PRAD",
                "dataset": "TCGA-PRAD",
                "include_mutation": True
            }
        }
    ]
    
    print(f"\n执行 {len(analysis_tasks)} 个批量分析任务...\n")
    
    all_results = []
    for i, task in enumerate(analysis_tasks, 1):
        print(f"[{i}/{len(analysis_tasks)}] 启动LLM分析: {task['name']}")
        
        result = system.run_analysis(
            research_goal=task['goal'],
            parameters=task.get('parameters', {})
        )
        
        all_results.append({
            "cancer_type": task['name'],
            "status": result['status'],
            "task_id": result['task_id'],
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"  ✓ 完成 (Task ID: {result['task_id']}, Status: {result['status']})\n")
    
    # 保存所有结果
    output_path = system.save_results(output_dir="./results/batch_analysis")
    
    print(f"\n" + "="*70)
    print(f"✓ 批量LLM分析完成")
    print(f"✓ {len(all_results)} 个任务已执行")
    print(f"✓ 结果已保存到: {output_path}")
    print(f"="*70)
    
    # 显示任务历史
    print(f"\n任务执行历史:")
    for task in system.get_task_history():
        print(f"  Task {task['id']}: {task['goal'][:40]}...")
        print(f"    - 状态: {task['status']}")
        print(f"    - 时间: {task.get('end_time', 'N/A')}\n")

    
    return all_results


def main():
    """
    运行所有生产级示例
    """
    print("\n" + "="*70)
    print("BioInfoMAS 生产级系统 - 完整工具调用演示")
    print("="*70)
    
    examples = [
        ("差异表达分析", example_differential_expression_analysis),
        ("变异检测分析", example_variant_calling_analysis),
        ("单细胞分析", example_single_cell_analysis),
        ("通路网络分析", example_pathway_and_network_analysis),
        ("批量分析", example_batch_analysis),
    ]
    
    print("\n选择要运行的示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples)+1}. 运行所有示例")
    print(f"  0. 退出")
    example_differential_expression_analysis()
    choice = input("\n请输入选择 (0-{0}): ".format(len(examples)+1)).strip()
    
    try:
        choice = int(choice)
        
        if choice == 0:
            print("退出程序")
            return
        
        elif 1 <= choice <= len(examples):
            name, example_func = examples[choice-1]
            print(f"\n正在运行: {name}...")
            example_func()
            
        elif choice == len(examples) + 1:
            print(f"\n正在运行所有 {len(examples)} 个示例...\n")
            for name, example_func in examples:
                try:
                    example_func()
                except Exception as e:
                    logger.error(f"示例 '{name}' 执行失败: {str(e)}", exc_info=True)
                
                print("\n按 Enter 继续下一个示例...\n")
                input()
        
        else:
            print("无效的选择")
    
    except ValueError:
        print("请输入有效的数字")


if __name__ == "__main__":
    main()
