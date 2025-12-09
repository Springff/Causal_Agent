"""
CausalAgent - Multi-Agent System for Bioinformatics Research
生物信息学多智能体系统
"""

__version__ = "1.0.0"
__author__ = "CausalAgent Team"
__all__ = ["CausalAgentProduction", "PlannerAgent", "DataProcessingAgent", "CausalFeatureSelectionAgent", 
           "FeatureScreeningAgent", "ValidationAgent"]

from .system_production import CausalAgentProduction
from .agents.DataProcessingAgent import DataProcessingAgent
from .agents.CausalFeatureSelectionAgent import CausalFeatureSelectionAgent
from .agents.FeatureScreeningAgent import FeatureScreeningAgent
from .agents.ValidationAgent import ValidationAgent
