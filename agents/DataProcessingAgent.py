"""
DataProcessingAgent - 数据获取与预处理智能体
根据数据概况，决定清洗策略和预处理方式
"""

from autogen import ConversableAgent
from typing import Dict, Any, List, Callable
import logging
import json

logger = logging.getLogger(__name__)


class DataProcessingAgent:
    """数据处理智能体"""

    SYSTEM_PROMPT = """# Role
你是一个数据预处理决策专家。你的任务是分析原始数据，并调用预定义的工具将数据转化为适合因果推断的格式。

# Constraints
- 每次回复只进行必要的思考和工具调用。
- 确保所有操作都不会改变变量的原始物理含义（不要做 PCA）。
"""

    def __init__(self, llm_config: Dict[str, Any]):
        """初始化数据智能体"""
        self.agent = ConversableAgent(
            name="DataProcessingAgent",
            system_message=self.SYSTEM_PROMPT,
            llm_config=llm_config,
            is_termination_msg=lambda x: "COMPLETE" in str(x.get("content", "")),
        )

    def get_agent(self) -> ConversableAgent:
        """获取AutoGen Agent对象"""
        return self.agent

    def register_tools(self, tools: List[Dict[str, Any]] = None):
        """为智能体注册数据处理工具：
        - 将可用工具写入系统提示，帮助模型决策
        - 把工具定义注册到 agent 的 LLM 配置注册
        """
        # 如果传入了全局工具定义则优先使用，否则回退到本地定义
        source_tools = tools if tools else self._get_tools_for_autogen()

        # 把可用工具写入系统提示
        try:
            tools_text = "\n".join(
                [f"- {t.get('function') or {}}" for t in source_tools]
            )
            tools_section = "\n\n可用工具（DataProcessingAgent）:\n" + (
                tools_text or "- 无可用工具"
            )
            new_prompt = self.SYSTEM_PROMPT + tools_section

            self.agent.update_system_message(new_prompt)
        except Exception:
            logger.debug("无法将工具写入 DataProcessingAgent 系统提示", exc_info=True)

        # 注册到 LLM
        try:
            if "tools" not in self.agent.llm_config:
                self.agent.llm_config["tools"] = []
            self.agent.llm_config["tools"].extend(source_tools)
        except Exception:
            logger.warning("为 DataProcessingAgent 注册工具时失败", exc_info=True)

        logger.info(f"✓ 为 DataProcessingAgent 注册 {len(source_tools)} 个数据工具")

    def _get_tools_for_autogen(self) -> List[Dict[str, Any]]:
        """
        获取所有工具定义，用于AutoGen的function_map
        返回工具函数列表供LLM调用
        - `get_data_summary(filepath)`: 获取数据行数、列数、缺失值比例、各列数据类型概览。
        - `impute_missing_values(strategy, method)`: 填补缺失值。参数 `strategy` 可选 ['drop', 'impute']，`method` 可选 ['mean', 'median', 'knn']。
        - `detect_variable_types(threshold)`: 自动识别变量是连续型还是离散型。
        - `discretize_column(column_name, bins)`: 将连续变量离散化（如果后续因果算法要求）。
        - `normalize_data(method)`: 标准化数据。
        """
        return [
            # 数据获取工具
            {
                "type": "function",
                "function": {
                    "name": "get_data_summary",
                    "description": "获取数据的基本概要信息，包括行数、列数、缺失值比例以及各列的数据类型",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "输入数据文件的路径（支持CSV、TSV等格式）",
                            }
                        },
                        "required": ["filepath"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "impute_missing_values",
                    "description": "根据指定策略和方法处理数据中的缺失值",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "strategy": {
                                "type": "string",
                                "enum": ["drop", "impute"],
                                "description": "缺失值处理策略：'drop' 删除含缺失值的行，'impute' 使用指定方法填补",
                            },
                            "method": {
                                "type": "string",
                                "enum": ["mean", "median", "knn"],
                                "description": "填补方法（仅在 strategy='impute' 时有效）：'mean' 使用均值，'median' 使用中位数，'knn' 使用K近邻插补",
                            },
                        },
                        "required": ["strategy"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_variable_types",
                    "description": "自动判断数据集中各列为连续型还是离散型变量",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "threshold": {
                                "type": "integer",
                                "description": "用于区分离散与连续变量的唯一值数量阈值（例如：唯一值少于该阈值视为离散型）",
                            }
                        },
                        "required": ["threshold"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "discretize_column",
                    "description": "将指定的连续变量列离散化为分箱区间",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column_name": {
                                "type": "string",
                                "description": "需要离散化的列名",
                            },
                            "bins": {
                                "type": "integer",
                                "description": "分箱数量（例如：5 表示将数据划分为5个区间）",
                            },
                        },
                        "required": ["column_name", "bins"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "normalize_data",
                    "description": "对数据进行标准化或归一化处理",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["z-score", "min-max", "robust"],
                                "description": "标准化方法：'z-score'（均值为0，标准差为1），'min-max'（缩放到[0,1]），'robust'（基于中位数和四分位距）",
                            }
                        },
                        "required": ["method"],
                    },
                },
            },
        ]
