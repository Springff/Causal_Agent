"""
BioInfoMAS - Multi-Agent System for Bioinformatics Research
生物信息学多智能体系统
"""

__version__ = "1.0.0"
__author__ = "BioInfoMAS Team"
__all__ = ["BioInfoMASProduction", "OrchestratorAgent", "DataAgent", "AnalysisAgent", 
           "KnowledgeAgent", "VisualizationAgent"]

from .system_production import BioInfoMASProduction
from .agents.orchestrator_agent import OrchestratorAgent
from .agents.data_agent import DataAgent
from .agents.analysis_agent import AnalysisAgent
from .agents.knowledge_agent import KnowledgeAgent
from .agents.visualization_agent import VisualizationAgent
