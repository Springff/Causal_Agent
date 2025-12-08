"""
FeatureScreeningAgent - 特征筛选智能体
利用统计相关性和LLM的语义知识，快速剔除无关变量，缩小搜索空间
"""

from autogen import ConversableAgent
from typing import Dict, Any, List, Callable
import logging
import json

logger = logging.getLogger(__name__)


class FeatureScreeningAgent:
    """特征筛选智能体"""

    SYSTEM_PROMPT = """# Role
你是一个结合了领域知识与统计能力的特征筛选专家。你的任务是缩小因果搜索的范围，剔除明显的噪声变量。

# Constraints
- **宁滥勿缺**：如果不确定某个变量是否有因果关系，请保留它。
- 在决定剔除某个变量时，必须简要说明理由（是基于统计得分低，还是基于语义逻辑不可能）。"""

    def __init__(self, llm_config: Dict[str, Any]):
        """初始化知识智能体"""
        self.agent = ConversableAgent(
            name="FeatureScreeningAgent",
            system_message=self.SYSTEM_PROMPT,
            llm_config=llm_config,
            is_termination_msg=lambda x: "COMPLETE" in str(x.get("content", "")),
        )

    def get_agent(self) -> ConversableAgent:
        """获取AutoGen Agent对象"""
        return self.agent

    def register_tools(self, tools: List[Dict[str, Any]] = None):
        """为智能体注册知识工具：筛选相关工具并写入系统提示，然后注册到模型层。"""
        source_tools = tools or []
        if not source_tools:
            source_tools = (
                self._get_tools_for_autogen()
                if hasattr(self, "_get_tools_for_autogen")
                else []
            )

        # 写入系统提示
        try:
            tools_text = "\n".join(
                [f"- {t.get('function') or {}}" for t in source_tools]
            )
            tools_section = "\n\n可用工具（FeatureScreeningAgent）:\n" + (
                tools_text or "- 无可用工具"
            )
            new_prompt = self.SYSTEM_PROMPT + tools_section

            self.agent.update_system_message(new_prompt)
        except Exception:
            logger.debug("无法将工具写入 FeatureScreeningAgent 系统提示", exc_info=True)

        # 注册到 LLM
        try:
            if "tools" not in self.agent.llm_config:
                self.agent.llm_config["tools"] = []
            self.agent.llm_config["tools"].extend(source_tools)
        except Exception:
            logger.warning("为 FeatureScreeningAgent 注册工具时失败", exc_info=True)

        logger.info(f"✓ 为 FeatureScreeningAgent 注册 {len(source_tools)} 个知识工具")

    def _get_tools_for_autogen(self) -> List[Dict[str, Any]]:
        """
        获取所有工具定义，用于AutoGen的function_map
        返回工具函数列表供LLM调用

        # Available Tools
        - `calculate_correlation(target_var, method)`: 计算所有特征与目标变量的相关系数。
        - `calculate_mutual_information(target_var)`: 计算互信息，捕捉非线性关系。
        - `drop_features(feature_list)`: 从数据集中移除指定的特征列表。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate_correlation",
                    "description": "计算数据集中所有特征与指定目标变量之间的相关系数",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_var": {
                                "type": "string",
                                "description": "目标变量的列名",
                            },
                            "method": {
                                "type": "string",
                                "enum": ["pearson", "spearman", "kendall"],
                                "description": "相关系数计算方法：'pearson'（线性相关），'spearman'（秩相关），'kendall'（序相关）",
                            },
                        },
                        "required": ["target_var", "method"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_mutual_information",
                    "description": "计算所有特征与目标变量之间的互信息，用于捕捉非线性依赖关系",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_var": {
                                "type": "string",
                                "description": "目标变量的列名",
                            }
                        },
                        "required": ["target_var"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "drop_features",
                    "description": "从当前数据集中移除指定的特征（列）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feature_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "要删除的特征名称列表",
                            }
                        },
                        "required": ["feature_list"],
                    },
                },
            },
        ]
