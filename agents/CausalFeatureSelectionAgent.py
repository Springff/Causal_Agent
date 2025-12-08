"""
CausalFeatureSelectionAgent - 因果特征选择智能体
负责逻辑推理和算法调度，寻找局部因果特征
"""

from autogen import ConversableAgent
from typing import Dict, Any, List, Callable
import logging
import json

logger = logging.getLogger(__name__)


class CausalFeatureSelectionAgent:
    """因果特征选择智能体"""

    SYSTEM_PROMPT = """# Role
你是一个因果推断专家。你的任务是利用基于约束的算法，寻找目标变量的马尔可夫毯（Markov Blanket）：即父节点、子节点和配偶节点。

# Constraints
- 必须区分“相关性”和“因果性”。工具会帮你做数学计算，但你要负责解释结果。
- 如果工具返回结果为空，尝试调大 `alpha` 值重新调用。"""

    def __init__(self, llm_config: Dict[str, Any]):
        """初始化分析智能体"""
        self.agent = ConversableAgent(
            name="CausalFeatureSelectionAgent",
            system_message=self.SYSTEM_PROMPT,
            llm_config=llm_config,
            is_termination_msg=lambda x: "COMPLETE" in str(x.get("content", "")),
        )

    def get_agent(self) -> ConversableAgent:
        """获取AutoGen Agent对象"""
        return self.agent

    def register_tools(self, tools: List[Dict[str, Any]] = None):
        """为智能体注册分析工具：筛选相关工具并注册，同时将工具列表写入系统提示。"""
        source_tools = tools if tools else self._get_tools_for_autogen()

        # 把可用工具写入系统提示
        try:
            tools_text = "\n".join(
                [f"- {t.get('function') or {}}" for t in source_tools]
            )
            tools_section = "\n\n可用工具（CausalFeatureSelectionAgent）:\n" + (
                tools_text or "- 无可用工具"
            )
            new_prompt = self.SYSTEM_PROMPT + tools_section

            self.agent.update_system_message(new_prompt)
        except Exception:
            logger.debug(
                "无法将工具写入 CausalFeatureSelectionAgent 系统提示", exc_info=True
            )

        # 注册到 LLM
        try:
            if "tools" not in self.agent.llm_config:
                self.agent.llm_config["tools"] = []
            self.agent.llm_config["tools"].extend(source_tools)
        except Exception:
            logger.warning(
                "为 CausalFeatureSelectionAgent 注册工具时失败", exc_info=True
            )

        logger.info(
            f"✓ 为 CausalFeatureSelectionAgent 注册 {len(source_tools)} 个分析工具"
        )

    def _get_tools_for_autogen(self) -> List[Dict[str, Any]]:
        """
        获取所有工具定义，用于AutoGen的function_map
        返回工具函数列表供LLM调用
        # Available Tools
        - `run_local_discovery_algorithm(algorithm, target_var, alpha)`: 运行局部因果发现算法。参数 `algorithm` 可选 ['MMPC', 'HITON-PC', 'IAMB']，`alpha` 为显著性水平（默认0.05）。
        - `test_conditional_independence(x, y, conditioning_set)`: 单独测试 X 和 Y 在给定 Z 的条件下是否独立（用于微调或验证）。
        - `orient_edges(skeleton_graph)`: 根据 V-结构规则尝试确定无向边的方向。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_local_discovery_algorithm",
                    "description": "运行局部因果发现算法，识别与目标变量直接相关的特征（马尔可夫毯或父/子节点）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "algorithm": {
                                "type": "string",
                                "enum": ["MMPC", "HITON-PC", "IAMB"],
                                "description": "局部因果发现算法名称：'MMPC'（最大最小父节点和子节点）、'HITON-PC'、'IAMB'（增量关联马尔可夫毯）",
                            },
                            "target_var": {
                                "type": "string",
                                "description": "目标变量的列名",
                            },
                            "alpha": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.05,
                                "description": "统计检验的显著性水平（默认0.05）",
                            },
                        },
                        "required": ["algorithm", "target_var"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_conditional_independence",
                    "description": "测试两个变量 X 和 Y 在给定条件集 Z 下是否条件独立",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "string", "description": "变量 X 的列名"},
                            "y": {"type": "string", "description": "变量 Y 的列名"},
                            "conditioning_set": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "条件变量集合（列名列表）",
                            },
                        },
                        "required": ["x", "y", "conditioning_set"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "orient_edges",
                    "description": "基于 V-结构规则（如无向图中的碰撞结构）对骨架图中的边进行方向判定",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skeleton_graph": {
                                "type": "object",
                                "description": "无向骨架图，通常以邻接表或边列表形式表示（例如 {'A': ['B', 'C'], 'B': ['A'], 'C': ['A']}）",
                            }
                        },
                        "required": ["skeleton_graph"],
                    },
                },
            },
        ]
