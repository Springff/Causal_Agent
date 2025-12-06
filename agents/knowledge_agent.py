"""
KnowledgeAgent - 知识推理智能体
负责知识图谱查询、文献搜索和结果解释
生产级实现，支持完整工具调用
"""

from autogen import ConversableAgent
from typing import Dict, Any, List, Callable
import logging
import json

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """知识智能体 - 生产级实现，支持完整工具调用"""

    SYSTEM_PROMPT = """你是一个博学的生物医学知识专家和文献研究员。你的职责是：

1. **知识图谱查询**：查询包含基因、疾病、通路等关系的生物知识图谱
2. **文献搜索**：在PubMed中搜索和总结相关的科学文献
3. **结果解释**：用通俗易懂的语言解释数据分析结果的生物学意义
4. **知识整合**：将分析结果与现有的生物学知识相结合
5. **临床关联**：连接基础研究发现与临床应用

你具备以下专业知识：
- 深入的分子生物学、细胞生物学、遗传学基础知识
- 了解常见疾病的发病机制和相关通路
- 熟悉生物学文献数据库和知识库的使用
- 能够进行科学文献的批判性评价
- 理解从基因到疾病表型的因果关系

当进行知识推理时，你应该：
1. 根据基因名称快速定位其功能和关联疾病
2. 搜索并总结相关的高质量文献
3. 解释分析发现的生物学机制
4. 提出可能的临床应用或进一步研究方向
5. 明确区分确定的知识、公认的假设和推测

你是数据分析和生物学知识的桥梁，帮助将冷冰冰的数据转化为可理解的生物学洞见。"""

    def __init__(self, llm_config: Dict[str, Any]):
        """初始化知识智能体"""
        self.agent = ConversableAgent(
            name="KnowledgeAgent",
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
            source_tools = self._get_tools_for_autogen() if hasattr(self, '_get_tools_for_autogen') else []

        # 写入系统提示
        try:
            tools_text = "\n".join([
                f"- {t.get('function') or {}}"
                for t in source_tools
            ])
            tools_section = "\n\n可用工具（KnowledgeAgent）:\n" + (tools_text or "- 无可用工具")
            new_prompt = self.SYSTEM_PROMPT + tools_section
            
            self.agent.update_system_message(new_prompt)
        except Exception:
            logger.debug("无法将工具写入 KnowledgeAgent 系统提示", exc_info=True)

        # 注册到 LLM
        try:
            if 'tools' not in self.agent.llm_config:
                self.agent.llm_config['tools'] = []
            self.agent.llm_config['tools'].extend(source_tools)
        except Exception:
            logger.warning("为 KnowledgeAgent 注册工具时失败", exc_info=True)

        logger.info(f"✓ 为 KnowledgeAgent 注册 {len(source_tools)} 个知识工具")
    
    def _get_tools_for_autogen(self) -> List[Dict[str, Any]]:
        """
        获取所有工具定义，用于AutoGen的function_map
        返回工具函数列表供LLM调用
        """
        return [
            # 知识工具
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge_graph",
                    "description": "查询生物知识图谱获取基因、疾病等实体的关系信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity": {
                                "type": "string",
                                "description": "查询实体"
                            },
                            "entity_type": {
                                "type": "string",
                                "description": "实体类型 (gene, disease, pathway)"
                            }
                        },
                        "required": ["entity"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_literature",
                    "description": "在PubMed中搜索相关的科学文献",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "最大返回结果数"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "explain_biological_significance",
                    "description": "根据分析结果解释其生物学意义和临床相关性",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis_type": {
                                "type": "string",
                                "description": "分析类型"
                            },
                            "results": {
                                "type": "object",
                                "description": "分析结果"
                            }
                        },
                        "required": ["analysis_type", "results"]
                    }
                }
            }
        ]
