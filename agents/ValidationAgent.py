"""
ValidationAgent - 验证智能体
调用预测模型工具和稳定性测试工具，对因果结构进行打分
"""

from autogen import ConversableAgent
from typing import Dict, Any, List, Callable
import logging
import json

logger = logging.getLogger(__name__)


class ValidationAgent:
    """验证智能体"""

    SYSTEM_PROMPT = """# Role
你是一个严格的验证裁判。你的任务是评估上一步发现的“局部因果结构”的质量和可靠性。

# Constraints
- 不要只看精度，要看“特征-精度性价比”。
"""

    def __init__(self, llm_config: Dict[str, Any]):
        """初始化可视化智能体"""
        self.agent = ConversableAgent(
            name="ValidationAgent",
            system_message=self.SYSTEM_PROMPT,
            llm_config=llm_config,
            is_termination_msg=lambda x: "COMPLETE" in str(x.get("content", "")),
        )

    def get_agent(self) -> ConversableAgent:
        """获取AutoGen Agent对象"""
        return self.agent

    def register_tools(self, tools: List[Dict[str, Any]] = None):
        """为智能体注册可视化工具：筛选、写入系统提示并注册到模型层。"""
        source_tools = tools or []
        if not source_tools and hasattr(self, "_get_tools_for_autogen"):
            source_tools = self._get_tools_for_autogen()

        # 写入系统提示
        try:
            tools_text = "\n".join(
                [f"- {t.get('function') or {}}" for t in source_tools]
            )
            tools_section = "\n\n可用工具（ValidationAgent）:\n" + (
                tools_text or "- 无可用工具"
            )
            new_prompt = self.SYSTEM_PROMPT + tools_section

            self.agent.update_system_message(new_prompt)
        except Exception:
            logger.debug("无法将工具写入 ValidationAgent 系统提示", exc_info=True)

        # 注册到 LLM
        try:
            if "tools" not in self.agent.llm_config:
                self.agent.llm_config["tools"] = []
            self.agent.llm_config["tools"].extend(source_tools)
        except Exception:
            logger.warning("为 ValidationAgent 注册工具时失败", exc_info=True)

        logger.info(f"✓ 为 ValidationAgent 注册 {len(source_tools)} 个可视化工具")

    def _get_tools_for_autogen(self) -> List[Dict[str, Any]]:
        """
        获取所有工具定义，用于AutoGen的function_map
        返回工具函数列表供LLM调用
        # Available Tools
        - `evaluate_predictive_power(feature_set, target_var, model_type)`: 使用选定的特征集训练预测模型（如 RandomForest），返回 AUC 或 R2 分数。
        - `compare_with_baseline(feature_set, target_var)`: 自动对比“仅使用因果特征”与“使用所有特征”的预测性能。
        - `run_stability_selection(target_var, algorithm, n_resamples)`: 在数据子集上多次运行算法，返回特征出现的频率（置信度）。

        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "evaluate_predictive_power",
                    "description": "使用指定的特征集和目标变量训练预测模型，并返回模型性能指标（分类任务返回AUC，回归任务返回R²）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feature_set": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "用于训练的特征名称列表",
                            },
                            "target_var": {
                                "type": "string",
                                "description": "目标变量的列名",
                            },
                            "model_type": {
                                "type": "string",
                                "enum": [
                                    "RandomForest",
                                    "LogisticRegression",
                                    "LinearRegression",
                                    "XGBoost",
                                ],
                                "description": "用于评估的预测模型类型",
                            },
                        },
                        "required": ["feature_set", "target_var", "model_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_with_baseline",
                    "description": "对比仅使用因果发现得到的特征与使用全部特征在预测目标变量时的性能差异",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feature_set": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "因果发现筛选出的特征列表",
                            },
                            "target_var": {
                                "type": "string",
                                "description": "目标变量的列名",
                            },
                        },
                        "required": ["feature_set", "target_var"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_stability_selection",
                    "description": "通过在多个数据子样本上重复运行因果发现算法，评估每个特征被选中的稳定性（频率）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_var": {
                                "type": "string",
                                "description": "目标变量的列名",
                            },
                            "algorithm": {
                                "type": "string",
                                "enum": ["MMPC", "HITON-PC", "IAMB"],
                                "description": "用于稳定性选择的局部因果发现算法",
                            },
                            "n_resamples": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "重采样（如自助采样）的次数，用于评估特征选择的稳定性（例如：100）",
                            },
                        },
                        "required": ["target_var", "algorithm", "n_resamples"],
                    },
                },
            },
        ]
