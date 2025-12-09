"""
PlannerAgent - 规划智能体
负责任务分解、工作流编排和多智能体协调
"""

from autogen import ConversableAgent
from typing import Dict, List, Any, Callable
import logging

logger = logging.getLogger(__name__)


class PlannerAgent:
    """规划智能体"""

    SYSTEM_PROMPT = """# Role
你是一个“局部因果结构学习”任务的总指挥官和首席科学家。你的手下有四个专家智能体（DataProcessingAgent、FeatureScreeningAgent、CausalFeatureSelectionAgent、ValidationAgent）。

# Objective
根据用户给定的数据集路径和目标变量，规划并执行一套完整的研究方案，最终输出经过验证的目标变量的局部因果图（马尔可夫毯）。你并不直接处理数据，而是通过调用智能体来完成任务。
"""

    def __init__(self, llm_config: Dict[str, Any]):
        """初始化规划智能体"""
        self.agent = ConversableAgent(
            name="PlannerAgent",
            system_message=self.SYSTEM_PROMPT,
            llm_config=llm_config,
            is_termination_msg=lambda x: "COMPLETE" in str(x.get("content", "")),
        )
        self.task_history = []

    def get_agent(self) -> ConversableAgent:
        """获取AutoGen Agent对象"""
        return self.agent
    
    def register_tools(self, tools: List[Dict[str, Any]] = None):
        """为智能体注册工具：将工具列表写入系统提示并注册函数映射"""
        source_tools = tools if tools else []
        try:
            tools_text = "\n".join([
                f"- {t.get('function') or {}}"
                for t in source_tools
            ])
            tools_section = "\n\n可用工具（Orchestrator）:\n" + (tools_text or "- 无可用工具")
            new_prompt = self.SYSTEM_PROMPT + tools_section
            
            self.agent.update_system_message(new_prompt)
        except Exception:
            logger.debug("无法将工具写入 PlannerAgent 系统提示", exc_info=True)

        try:
            if 'tools' not in self.agent.llm_config:
                self.agent.llm_config['tools'] = []
            self.agent.llm_config['tools'].extend(source_tools)
        except Exception:
            logger.warning("为 PlannerAgent 注册工具时失败", exc_info=True)

        logger.info(f"✓ 为 PlannerAgent 注册 {len(tools or [])} 个工具")
    