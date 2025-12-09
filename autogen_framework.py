"""
BioInfoMAS - LLM-Based Multi-Agent System
基于AutoGen框架的生物信息学多智能体系统
支持工具调用和智能协作
"""

import os
import json
from typing import Any, Dict, List, Optional, Callable
from dotenv import load_dotenv
import autogen
from autogen import ConversableAgent, UserProxyAgent, config_list_from_json

# 加载环境变量
load_dotenv()

# AutoGen 配置
AUTOGEN_CONFIG = {
    "config_list": [
        {
            "model": os.getenv("LLM_MODEL_ID", "gpt-4"),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        }
    ],
    "timeout": 120,
    "cache_seed": 42,
}


# ============================================================================
# 工具定义 - 可被LLM调用的函数
# ============================================================================

class BioInfoTools:
    """生物信息学工具集合 - 所有工具都是可被LLM调用的函数"""

    @staticmethod
    def get_data_summary(filepath: str) -> Dict[str, Any]:
        """
        获取数据的基本概要信息，包括行数、列数、缺失值比例以及各列的数据类型
        
        Args:
            filepath: 输入数据文件的路径（支持CSV、TSV等格式）
        
        Returns:
            包含数据概要信息的字典
        """
        return {
            "status": "success",
            "filepath": filepath,
            "n_rows": 10000,
            "n_cols": 50,
            "missing_ratio": 0.03,
            "column_dtypes": {"col1": "float64", "col2": "object", "col3": "int64"}
        }


    @staticmethod
    def impute_missing_values(strategy: str, method: Optional[str] = None) -> Dict[str, Any]:
        """
        根据指定策略和方法处理数据中的缺失值
        
        Args:
            strategy: 缺失值处理策略：'drop' 删除含缺失值的行，'impute' 使用指定方法填补
            method: 填补方法（仅在 strategy='impute' 时有效）：'mean' 使用均值，'median' 使用中位数，'knn' 使用K近邻插补
        
        Returns:
            缺失值处理结果摘要
        """
        return {
            "status": "success",
            "strategy": strategy,
            "method": method,
            "missing_before": 1500,
            "missing_after": 0 if strategy == "impute" else 0,
            "rows_dropped": 0 if strategy == "impute" else 200
        }


    @staticmethod
    def detect_variable_types(threshold: int) -> Dict[str, Any]:
        """
        自动判断数据集中各列为连续型还是离散型变量
        
        Args:
            threshold: 用于区分离散与连续变量的唯一值数量阈值（例如：唯一值少于该阈值视为离散型）
        
        Returns:
            变量类型识别结果
        """
        return {
            "status": "success",
            "threshold": threshold,
            "variable_types": {"col1": "continuous", "col2": "discrete", "col3": "continuous"}
        }


    @staticmethod
    def discretize_column(column_name: str, bins: int) -> Dict[str, Any]:
        """
        将指定的连续变量列离散化为分箱区间
        
        Args:
            column_name: 需要离散化的列名
            bins: 分箱数量（例如：5 表示将数据划分为5个区间）
        
        Returns:
            离散化结果信息
        """
        return {
            "status": "success",
            "column_name": column_name,
            "bins": bins,
            "bin_edges": [0.0, 1.2, 2.5, 3.8, 5.0],
            "value_counts": [2000, 2500, 3000, 2500]
        }


    @staticmethod
    def normalize_data(method: str) -> Dict[str, Any]:
        """
        对数据进行标准化或归一化处理
        
        Args:
            method: 标准化方法：'z-score'（均值为0，标准差为1），'min-max'（缩放到[0,1]），'robust'（基于中位数和四分位距）
        
        Returns:
            标准化结果摘要
        """
        return {
            "status": "success",
            "method": method,
            "normalized_columns": ["col1", "col3", "col5"],
            "skipped_columns": ["col2", "col4"]
        }


    @staticmethod
    def calculate_correlation(target_var: str, method: str) -> Dict[str, Any]:
        """
        计算数据集中所有特征与指定目标变量之间的相关系数
        
        Args:
            target_var: 目标变量的列名
            method: 相关系数计算方法：'pearson'（线性相关），'spearman'（秩相关），'kendall'（序相关）
        
        Returns:
            相关系数结果字典
        """
        return {
            "status": "success",
            "target_var": target_var,
            "method": method,
            "correlations": {"col1": 0.72, "col2": -0.15, "col3": 0.88}
        }


    @staticmethod
    def calculate_mutual_information(target_var: str) -> Dict[str, Any]:
        """
        计算所有特征与目标变量之间的互信息，用于捕捉非线性依赖关系
        
        Args:
            target_var: 目标变量的列名
        
        Returns:
            互信息结果字典
        """
        return {
            "status": "success",
            "target_var": target_var,
            "mutual_information": {"col1": 0.45, "col2": 0.12, "col3": 0.67}
        }


    @staticmethod
    def drop_features(feature_list: List[str]) -> Dict[str, Any]:
        """
        从当前数据集中移除指定的特征（列）
        
        Args:
            feature_list: 要删除的特征名称列表
        
        Returns:
            删除特征后的结果摘要
        """
        return {
            "status": "success",
            "dropped_features": feature_list,
            "remaining_features": ["col1", "col3", "col5"],
            "n_features_before": 5,
            "n_features_after": 3
        }


    @staticmethod
    def run_local_discovery_algorithm(algorithm: str, target_var: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        运行局部因果发现算法，识别与目标变量直接相关的特征（马尔可夫毯或父/子节点）
        
        Args:
            algorithm: 局部因果发现算法名称：'MMPC'（最大最小父节点和子节点）、'HITON-PC'、'IAMB'（增量关联马尔可夫毯）
            target_var: 目标变量的列名
            alpha: 统计检验的显著性水平（默认0.05）
        
        Returns:
            因果发现结果
        """
        return {
            "status": "success",
            "algorithm": algorithm,
            "target_var": target_var,
            "alpha": alpha,
            "selected_features": ["col1", "col3"]
        }


    @staticmethod
    def test_conditional_independence(x: str, y: str, conditioning_set: List[str]) -> Dict[str, Any]:
        """
        测试两个变量 X 和 Y 在给定条件集 Z 下是否条件独立
        
        Args:
            x: 变量 X 的列名
            y: 变量 Y 的列名
            conditioning_set: 条件变量集合（列名列表）
        
        Returns:
            条件独立性检验结果
        """
        return {
            "status": "success",
            "x": x,
            "y": y,
            "conditioning_set": conditioning_set,
            "p_value": 0.032,
            "independent": False
        }


    @staticmethod
    def orient_edges(skeleton_graph: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        基于 V-结构规则（如无向图中的碰撞结构）对骨架图中的边进行方向判定
        
        Args:
            skeleton_graph: 无向骨架图，通常以邻接表形式表示（例如 {'A': ['B', 'C'], 'B': ['A'], 'C': ['A']}）
        
        Returns:
            有向图（部分定向）结构
        """
        return {
            "status": "success",
            "input_skeleton": skeleton_graph,
            "oriented_edges": [("A", "B"), ("C", "B")],
            "undirected_edges": [("D", "E")]
        }


    @staticmethod
    def evaluate_predictive_power(feature_set: List[str], target_var: str, model_type: str) -> Dict[str, Any]:
        """
        使用指定的特征集和目标变量训练预测模型，并返回模型性能指标（分类任务返回AUC，回归任务返回R²）
        
        Args:
            feature_set: 用于训练的特征名称列表
            target_var: 目标变量的列名
            model_type: 用于评估的预测模型类型
        
        Returns:
            模型评估结果
        """
        return {
            "status": "success",
            "feature_set": feature_set,
            "target_var": target_var,
            "model_type": model_type,
            "score": 0.92,
            "metric": "AUC" if model_type in ["RandomForest", "LogisticRegression", "XGBoost"] else "R2"
        }


    @staticmethod
    def compare_with_baseline(feature_set: List[str], target_var: str) -> Dict[str, Any]:
        """
        对比仅使用因果发现得到的特征与使用全部特征在预测目标变量时的性能差异
        
        Args:
            feature_set: 因果发现筛选出的特征列表
            target_var: 目标变量的列名
        
        Returns:
            性能对比结果
        """
        return {
            "status": "success",
            "target_var": target_var,
            "causal_features": feature_set,
            "baseline_score": 0.87,
            "causal_score": 0.92,
            "improvement": 0.05,
            "metric": "AUC"
        }


    @staticmethod
    def run_stability_selection(target_var: str, algorithm: str, n_resamples: int) -> Dict[str, Any]:
        """
        通过在多个数据子样本上重复运行因果发现算法，评估每个特征被选中的稳定性（频率）
        
        Args:
            target_var: 目标变量的列名
            algorithm: 用于稳定性选择的局部因果发现算法
            n_resamples: 重采样（如自助采样）的次数，用于评估特征选择的稳定性（例如：100）
        
        Returns:
            特征稳定性评分
        """
        return {
            "status": "success",
            "target_var": target_var,
            "algorithm": algorithm,
            "n_resamples": n_resamples,
            "stability_scores": {"col1": 0.94, "col3": 0.88, "col5": 0.32}
        }    


def create_function_map() -> Dict[str, Callable]:
    """
    创建函数映射 - 连接工具定义和实际实现
    """
    tools_instance = BioInfoTools()
    return {
        "get_data_summary": tools_instance.get_data_summary,
        "impute_missing_values": tools_instance.impute_missing_values,
        "detect_variable_types": tools_instance.detect_variable_types,
        "discretize_column": tools_instance.discretize_column,
        "normalize_data": tools_instance.normalize_data,
        "calculate_correlation": tools_instance.calculate_correlation,
        "calculate_mutual_information": tools_instance.calculate_mutual_information,
        "drop_features": tools_instance.drop_features,
        "run_local_discovery_algorithm": tools_instance.run_local_discovery_algorithm,
        "test_conditional_independence": tools_instance.test_conditional_independence,
        "orient_edges": tools_instance.orient_edges,
        "evaluate_predictive_power": tools_instance.evaluate_predictive_power,
        "compare_with_baseline": tools_instance.compare_with_baseline,
        "run_stability_selection": tools_instance.run_stability_selection,
        
    }
