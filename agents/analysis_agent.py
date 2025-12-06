"""
AnalysisAgent - 数据分析智能体
负责运行各种生物信息学分析（差异表达、变异检测、通路富集等）
生产级实现，支持完整工具调用
"""

from autogen import ConversableAgent
from typing import Dict, Any, List, Callable
import logging
import json

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """分析智能体 - 生产级实现，支持完整工具调用"""

    SYSTEM_PROMPT = """你是一个高度专业的计算生物学分析专家。你的职责是：

1. **差异表达分析**：使用DESeq2、edgeR等方法识别不同条件下差异表达的基因
2. **变异检测**：识别SNP、插入缺失和结构变异
3. **通路富集分析**：使用GO、KEGG、Reactome数据库进行功能富集分析
4. **序列相似性搜索**：使用BLAST进行同源序列搜索
5. **序列比对**：执行多序列比对和进化分析
6. **统计分析**：应用适当的统计方法进行假设检验

你具备以下专业知识：
- 深入理解各种生物信息学分析方法的原理和适用场景
- 掌握统计学基础，能够正确解释p值、q值等统计指标
- 熟悉常用的分析工具和软件参数
- 能够选择合适的分析方法处理不同类型的数据

当执行分析任务时，你应该：
1. 评估输入数据的质量和特性
2. 选择最适合的分析方法
3. 设置合理的统计阈值和参数
4. 详细解释分析结果
5. 提供统计学支持和置信度评估

你是发现生物学规律的引擎，通过严谨的分析揭示隐藏在数据中的生物学真相。"""

    def __init__(self, llm_config: Dict[str, Any]):
        """初始化分析智能体"""
        self.agent = ConversableAgent(
            name="AnalysisAgent",
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
            tools_text = "\n".join([
                f"- {t.get('function') or {}}"
                for t in source_tools
            ])
            tools_section = "\n\n可用工具（AnalysisAgent）:\n" + (tools_text or "- 无可用工具")
            new_prompt = self.SYSTEM_PROMPT + tools_section

            self.agent.update_system_message(new_prompt)
        except Exception:
            logger.debug("无法将工具写入 AnalysisAgent 系统提示", exc_info=True)

        # 注册到 LLM
        try:
            if 'tools' not in self.agent.llm_config:
                self.agent.llm_config['tools'] = []
            self.agent.llm_config['tools'].extend(source_tools)
        except Exception:
            logger.warning("为 AnalysisAgent 注册工具时失败", exc_info=True)

        logger.info(f"✓ 为 AnalysisAgent 注册 {len(source_tools)} 个分析工具")
    
   
    def _get_tools_for_autogen(self) -> List[Dict[str, Any]]:
        """
        获取所有工具定义，用于AutoGen的function_map
        返回工具函数列表供LLM调用
        """
        return [
            # 数据分析工具
            {
                "type": "function",
                "function": {
                    "name": "differential_expression_analysis",
                    "description": "执行差异表达分析（DESeq2或edgeR）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "count_matrix": {
                                "type": "string",
                                "description": "基因计数矩阵文件"
                            },
                            "control_samples": {
                                "type": "array",
                                "description": "对照组样本列表"
                            },
                            "treatment_samples": {
                                "type": "array",
                                "description": "处理组样本列表"
                            },
                            "method": {
                                "type": "string",
                                "description": "分析方法 (deseq2, edger)"
                            }
                        },
                        "required": ["count_matrix", "control_samples", "treatment_samples"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "pathway_enrichment",
                    "description": "执行通路富集分析（GO、KEGG等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "gene_list": {
                                "type": "array",
                                "description": "基因列表"
                            },
                            "database": {
                                "type": "string",
                                "description": "数据库 (kegg, go, reactome)"
                            }
                        },
                        "required": ["gene_list", "database"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "variant_calling",
                    "description": "执行变异检测（SNP、indel等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bam_file": {
                                "type": "string",
                                "description": "BAM文件路径"
                            },
                            "reference_genome": {
                                "type": "string",
                                "description": "参考基因组"
                            }
                        },
                        "required": ["bam_file", "reference_genome"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sequence_blast",
                    "description": "执行BLAST序列相似性搜索",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_sequence": {
                                "type": "string",
                                "description": "查询序列"
                            },
                            "database": {
                                "type": "string",
                                "description": "BLAST数据库"
                            }
                        },
                        "required": ["query_sequence"]
                    }
                }
            },
             ]


    # def create_function_map() -> Dict[str, Callable]:
    #     """
    #     创建函数映射 - 连接工具定义和实际实现
    #     """
    #     tools_instance = BioInfoTools()
    #     return {
    #         "download_data": tools_instance.download_data,
    #         "quality_control": tools_instance.quality_control,
    #         "preprocess_data": tools_instance.preprocess_data,
    #         "differential_expression_analysis": tools_instance.differential_expression_analysis,
    #         "pathway_enrichment": tools_instance.pathway_enrichment,
    #         "variant_calling": tools_instance.variant_calling,
    #         "sequence_blast": tools_instance.sequence_blast,
    #         "query_knowledge_graph": tools_instance.query_knowledge_graph,
    #         "search_literature": tools_instance.search_literature,
    #         "explain_biological_significance": tools_instance.explain_biological_significance,
    #         "generate_volcano_plot": tools_instance.generate_volcano_plot,
    #         "generate_heatmap": tools_instance.generate_heatmap,
    #         "generate_pca_plot": tools_instance.generate_pca_plot,
    #         "generate_report": tools_instance.generate_report,
    #     }
