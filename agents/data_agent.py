"""
DataAgent - 数据获取与预处理智能体
负责数据下载、细胞图建模与标志性子图提取等
支持工具调用
"""

from autogen import ConversableAgent
from typing import Dict, Any, List, Callable
import logging
import json

logger = logging.getLogger(__name__)


class DataAgent:
    """数据处理智能体，支持完整工具调用"""

    SYSTEM_PROMPT = """你是一个专业的生物信息学数据处理专家。你的职责是：

1. **数据获取**：从GEO、TCGA、NCBI等公共数据库下载原始数据
2. **建模细胞图**：根据空间转录组数据中细胞位置构建细胞图
3. **提取代表性子图**：如果细胞图过大，提取具有代表性的子图进行后续分析


你具备以下专业知识：
- 对不同生物学数据类型的理解：转录组数据、基因组数据、蛋白质组数据等
- 常见生物信息学工具的参数和用法
- 处理大规模高通量测序数据的经验

当需要你处理数据时，你应该：
1. 确认数据来源和类型
2. 制定数据处理方案
3. 构建细胞图，了解图的基本属性
4. 当图中节点数超过100时，提取代表性子图

你是数据流水线的守门人，确保进入分析的数据质量可靠。"""

    def __init__(self, llm_config: Dict[str, Any]):
        """初始化数据智能体"""
        self.agent = ConversableAgent(
            name="DataAgent",
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
            tools_text = "\n".join([
                f"- {t.get('function') or {}}"
                for t in source_tools
            ])
            tools_section = "\n\n可用工具（DataAgent）:\n" + (tools_text or "- 无可用工具")
            new_prompt = self.SYSTEM_PROMPT + tools_section

            self.agent.update_system_message(new_prompt)
        except Exception:
            logger.debug("无法将工具写入 DataAgent 系统提示", exc_info=True)

        # 注册到 LLM
        try:
            if 'tools' not in self.agent.llm_config:
                self.agent.llm_config['tools'] = []
            self.agent.llm_config['tools'].extend(source_tools)
        except Exception:
            logger.warning("为 DataAgent 注册工具时失败", exc_info=True)

        logger.info(f"✓ 为 DataAgent 注册 {len(source_tools)} 个数据工具")
 
    def _get_tools_for_autogen(self) -> List[Dict[str, Any]]:
        """
        获取所有工具定义，用于AutoGen的function_map
        返回工具函数列表供LLM调用
        """
        return [
            # 数据获取工具
            {
                "type": "function",
                "function": {
                    "name": "download_raw_data",
                    "description": "从公共生物数据库（GEO, TCGA, NCBI等）下载原始测序数据或临床数据",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "description": "目标数据库名称"
                            },
                            "accession_id": {
                                "type": "string",
                                "description": "数据的唯一访问编号"
                            },
                            "data_type": {
                                "type": "string",
                                "description": "下载的数据文件格式"
                            },
                            "output_dir": {
                                "type": "string",
                                "description": "数据保存的本地路径"
                            }
                        },
                        "required": ["database", "accession_id", "data_type", "output_dir"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "construct_cell_graph",
                    "description": "根据空间转录组数据中的细胞物理坐标构建空间邻域图（细胞图）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spatial_data_path": {
                                "type": "string",
                                "description": "包含细胞基因表达矩阵和空间坐标的输入文件路径"
                            },
                            "method": {
                                "type": "string",
                                "enum": ["knn", "radius", "delaunay"],
                                "description": "构建图的算法：K近邻(knn)、固定半径(radius)或Delaunay三角剖分"
                            },
                            "k_neighbors": {
                                "type": "integer",
                                "description": "如果使用KNN算法，指定每个细胞的邻居数量 (例如: 6)"
                            },
                            "radius_cutoff": {
                                "type": "number",
                                "description": "如果使用Radius算法，指定连接细胞的最大欧氏距离"
                            }
                        },
                        "required": ["spatial_data_path", "method"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_representative_subgraphs",
                    "description": "针对过大的细胞图，通过采样策略提取具有代表性的子图以降低计算复杂度",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input_graph_data": {
                                "type": "string",
                                "description": "已构建的完整大图数据对象或文件路径"
                            },
                            "sampling_strategy": {
                                "type": "string",
                                "enum": ["random_walk", "node_sampling", "community_based"],
                                "description": "提取子图的采样策略 (例如: 随机游走, 节点采样, 基于社区检测)"
                            },
                            "subgraph_size": {
                                "type": "integer",
                                "description": "每个子图包含的目标节点（细胞）数量"
                            },
                            "num_subgraphs": {
                                "type": "integer",
                                "description": "需要提取的子图总数量"
                            }
                        },
                        "required": ["input_graph_data"]
                    }
                }
            }
        ]

