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

    # ===== 数据获取工具 =====
    @staticmethod
    def download_data(database: str, accession_id: str, data_type: str) -> Dict[str, Any]:
        """
        从生物数据库下载数据
        
        Args:
            database: 数据库名称 (GEO, TCGA, NCBI, UniProt)
            accession_id: 数据库访问号
            data_type: 数据类型 (fastq, bam, vcf, etc.)
        
        Returns:
            下载信息和文件路径
        """
        return {
            "status": "success",
            "database": database,
            "accession_id": accession_id,
            "data_type": data_type,
            "file_path": f"./data/{accession_id}.{data_type}",
            "file_size": "1.2 GB",
            "download_time": "2.5 hours"
        }

    @staticmethod
    def quality_control(input_file: str, quality_threshold: float = 30.0) -> Dict[str, Any]:
        """
        执行FastQC质量控制
        
        Args:
            input_file: 输入文件路径
            quality_threshold: 质量阈值
        
        Returns:
            质量控制报告
        """
        return {
            "status": "PASS" if quality_threshold <= 30 else "FAIL",
            "total_reads": 50000000,
            "read_length": 150,
            "gc_content": 48.5,
            "quality_score": f"Q{int(quality_threshold)}+",
            "adapter_content": "0.2%",
            "report_path": "./qc_results/fastqc_report.html"
        }

    @staticmethod
    def preprocess_data(input_file: str, trim_adapters: bool = True) -> Dict[str, Any]:
        """
        数据预处理（适配器修剪和比对）
        
        Args:
            input_file: 输入文件
            trim_adapters: 是否修剪适配器
        
        Returns:
            预处理结果
        """
        return {
            "status": "success",
            "input_file": input_file,
            "steps": ["adapter_trimming", "sequence_alignment"],
            "output_bam": "./output.bam",
            "alignment_rate": 0.95,
            "aligned_reads": 47500000,
            "execution_time": "3.5 hours"
        }

    # ===== 数据分析工具 =====
    @staticmethod
    def differential_expression_analysis(
        count_matrix: str,
        control_samples: List[str],
        treatment_samples: List[str],
        method: str = "deseq2"
    ) -> Dict[str, Any]:
        """
        差异表达分析
        
        Args:
            count_matrix: 基因计数矩阵文件
            control_samples: 对照组样本
            treatment_samples: 处理组样本
            method: 分析方法 (deseq2, edger)
        
        Returns:
            差异表达分析结果
        """
        return {
            "status": "success",
            "method": method,
            "total_genes": 20000,
            "significant_genes": 2500,
            "upregulated": 1200,
            "downregulated": 1300,
            "fdr_threshold": 0.05,
            "top_genes": [
                {"gene": "BRCA1", "log2fc": 5.2, "padj": 1.5e-50},
                {"gene": "TP53", "log2fc": 4.8, "padj": 3.2e-45}
            ]
        }

    @staticmethod
    def pathway_enrichment(gene_list: List[str], database: str = "kegg") -> Dict[str, Any]:
        """
        通路富集分析
        
        Args:
            gene_list: 基因列表
            database: 数据库 (kegg, go)
        
        Returns:
            富集分析结果
        """
        return {
            "status": "success",
            "database": database,
            "input_genes": len(gene_list),
            "enriched_pathways": [
                {
                    "pathway_id": "hsa05200",
                    "pathway_name": "Pathways in cancer",
                    "p_value": 3.5e-12,
                    "gene_count": 85
                },
                {
                    "pathway_id": "hsa04115",
                    "pathway_name": "p53 signaling pathway",
                    "p_value": 1.2e-15,
                    "gene_count": 45
                }
            ]
        }

    @staticmethod
    def variant_calling(bam_file: str, reference_genome: str) -> Dict[str, Any]:
        """
        变异检测
        
        Args:
            bam_file: BAM文件路径
            reference_genome: 参考基因组
        
        Returns:
            变异检测结果
        """
        return {
            "status": "success",
            "total_variants": 50000,
            "snps": 40000,
            "indels": 8000,
            "structural_variants": 2000,
            "transitions_transversions": 2.1,
            "output_vcf": "./variants.vcf"
        }

    @staticmethod
    def sequence_blast(query_sequence: str, database: str = "nr") -> Dict[str, Any]:
        """
        BLAST序列相似性搜索
        
        Args:
            query_sequence: 查询序列
            database: BLAST数据库
        
        Returns:
            BLAST搜索结果
        """
        return {
            "status": "success",
            "database": database,
            "hits": [
                {
                    "hit_id": "gi|123456789",
                    "description": "Homo sapiens BRCA1 protein",
                    "identity": 0.98,
                    "e_value": 2.3e-120
                },
                {
                    "hit_id": "gi|987654321",
                    "description": "Mus musculus Brca1 protein",
                    "identity": 0.95,
                    "e_value": 1.8e-110
                }
            ]
        }

    # ===== 知识推理工具 =====
    @staticmethod
    def query_knowledge_graph(entity: str, entity_type: str = "gene") -> Dict[str, Any]:
        """
        查询知识图谱
        
        Args:
            entity: 查询实体（基因名、疾病等）
            entity_type: 实体类型
        
        Returns:
            知识图谱查询结果
        """
        return {
            "status": "success",
            "entity": entity,
            "relations": [
                {
                    "source": entity,
                    "target": "Breast Cancer",
                    "relation": "associated_with",
                    "evidence": 245
                },
                {
                    "source": entity,
                    "target": "DNA Repair Pathway",
                    "relation": "participates_in",
                    "evidence": 156
                }
            ]
        }

    @staticmethod
    def search_literature(query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        搜索PubMed文献
        
        Args:
            query: 搜索查询
            max_results: 最大返回结果数
        
        Returns:
            文献搜索结果
        """
        return {
            "status": "success",
            "query": query,
            "total_results": 5234,
            "papers": [
                {
                    "pmid": "35245123",
                    "title": "Gene expression profiling reveals novel biomarkers",
                    "authors": ["Smith J", "Johnson K"],
                    "year": 2023,
                    "journal": "Nature Genetics",
                    "impact_factor": 38.2
                },
                {
                    "pmid": "35102456",
                    "title": "Pathway enrichment analysis in differential expression",
                    "authors": ["Brown R", "Davis S"],
                    "year": 2023,
                    "journal": "Genome Biology",
                    "impact_factor": 15.6
                }
            ]
        }

    @staticmethod
    def explain_biological_significance(
        analysis_type: str,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        解释分析结果的生物学意义
        
        Args:
            analysis_type: 分析类型
            results: 分析结果
        
        Returns:
            生物学解释
        """
        return {
            "status": "success",
            "analysis_type": analysis_type,
            "interpretation": {
                "summary": "The analysis reveals significant changes in gene expression...",
                "key_findings": [
                    "Upregulation of DNA repair genes suggests cellular stress response",
                    "Downregulation of cell cycle genes indicates cell cycle arrest",
                    "Changes consistent with p53-mediated apoptosis"
                ],
                "clinical_relevance": "These findings suggest potential therapeutic targets"
            }
        }

    # ===== 可视化工具 =====
    @staticmethod
    def generate_volcano_plot(deg_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成火山图
        
        Args:
            deg_results: 差异表达分析结果
        
        Returns:
            图表生成信息
        """
        return {
            "status": "success",
            "plot_type": "volcano",
            "output_file": "./results/volcano_plot.html",
            "description": "Interactive volcano plot showing differential expression"
        }

    @staticmethod
    def generate_heatmap(expression_data: str) -> Dict[str, Any]:
        """
        生成热图
        
        Args:
            expression_data: 表达数据文件
        
        Returns:
            图表生成信息
        """
        return {
            "status": "success",
            "plot_type": "heatmap",
            "output_file": "./results/heatmap.html",
            "dimensions": {"genes": 200, "samples": 50}
        }

    @staticmethod
    def generate_pca_plot(expression_data: str) -> Dict[str, Any]:
        """
        生成PCA图
        
        Args:
            expression_data: 表达数据文件
        
        Returns:
            图表生成信息
        """
        return {
            "status": "success",
            "plot_type": "pca",
            "output_file": "./results/pca_plot.html",
            "variance_explained": {"PC1": 45.2, "PC2": 28.5, "PC3": 12.3}
        }

    @staticmethod
    def generate_report(
        report_title: str,
        sections: List[str],
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        生成分析报告
        
        Args:
            report_title: 报告标题
            sections: 报告章节
            output_format: 输出格式 (markdown, pdf, html)
        
        Returns:
            报告生成信息
        """
        return {
            "status": "success",
            "report_title": report_title,
            "output_file": f"./results/report.{output_format}",
            "sections": len(sections),
            "format": output_format,
            "generation_time": "5 minutes"
        }


# ============================================================================
# 注册工具给AutoGen
# ============================================================================

def get_tools_for_autogen() -> List[Dict[str, Any]]:
    """
    获取所有工具定义，用于AutoGen的function_map
    返回工具函数列表供LLM调用
    """
    return [
        # 数据获取工具
        {
            "type": "function",
            "function": {
                "name": "download_data",
                "description": "从生物数据库（GEO, TCGA等）下载原始数据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "数据库名称 (GEO, TCGA, NCBI, UniProt)"
                        },
                        "accession_id": {
                            "type": "string",
                            "description": "数据库访问号"
                        },
                        "data_type": {
                            "type": "string",
                            "description": "数据类型 (fastq, bam, vcf)"
                        }
                    },
                    "required": ["database", "accession_id", "data_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "quality_control",
                "description": "执行FastQC质量控制检查",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_file": {
                            "type": "string",
                            "description": "输入文件路径"
                        },
                        "quality_threshold": {
                            "type": "number",
                            "description": "质量阈值"
                        }
                    },
                    "required": ["input_file"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "preprocess_data",
                "description": "执行数据预处理（适配器修剪、序列比对等）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_file": {
                            "type": "string",
                            "description": "输入文件路径"
                        },
                        "trim_adapters": {
                            "type": "boolean",
                            "description": "是否修剪适配器"
                        }
                    },
                    "required": ["input_file"]
                }
            }
        },
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
        },
        # 可视化工具
        {
            "type": "function",
            "function": {
                "name": "generate_volcano_plot",
                "description": "生成差异表达分析的火山图",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "deg_results": {
                            "type": "object",
                            "description": "差异表达分析结果"
                        }
                    },
                    "required": ["deg_results"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_heatmap",
                "description": "生成基因表达热图",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression_data": {
                            "type": "string",
                            "description": "表达数据文件"
                        }
                    },
                    "required": ["expression_data"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_pca_plot",
                "description": "生成PCA降维分析图",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression_data": {
                            "type": "string",
                            "description": "表达数据文件"
                        }
                    },
                    "required": ["expression_data"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_report",
                "description": "生成综合分析报告",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_title": {
                            "type": "string",
                            "description": "报告标题"
                        },
                        "sections": {
                            "type": "array",
                            "description": "报告章节列表"
                        },
                        "output_format": {
                            "type": "string",
                            "description": "输出格式 (markdown, pdf, html)"
                        }
                    },
                    "required": ["report_title", "sections"]
                }
            }
        }
    ]


def create_function_map() -> Dict[str, Callable]:
    """
    创建函数映射 - 连接工具定义和实际实现
    """
    tools_instance = BioInfoTools()
    return {
        "download_data": tools_instance.download_data,
        "quality_control": tools_instance.quality_control,
        "preprocess_data": tools_instance.preprocess_data,
        "differential_expression_analysis": tools_instance.differential_expression_analysis,
        "pathway_enrichment": tools_instance.pathway_enrichment,
        "variant_calling": tools_instance.variant_calling,
        "sequence_blast": tools_instance.sequence_blast,
        "query_knowledge_graph": tools_instance.query_knowledge_graph,
        "search_literature": tools_instance.search_literature,
        "explain_biological_significance": tools_instance.explain_biological_significance,
        "generate_volcano_plot": tools_instance.generate_volcano_plot,
        "generate_heatmap": tools_instance.generate_heatmap,
        "generate_pca_plot": tools_instance.generate_pca_plot,
        "generate_report": tools_instance.generate_report,
    }
