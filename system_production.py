"""
BioInfoMAS - 生产级多智能体系统
具有完整工具调用能力的LLM驱动生物信息学分析平台
"""

import os
import json
import logging
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
from dotenv import load_dotenv

import autogen
from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager

from agents.PlannerAgent import PlannerAgent
from agents.DataProcessingAgent import DataProcessingAgent
from agents.CausalFeatureSelectionAgent import CausalFeatureSelectionAgent
from agents.FeatureScreeningAgent import FeatureScreeningAgent
from agents.ValidationAgent import ValidationAgent
from autogen_framework import create_function_map

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class CausalAgentProduction:
    """
    生产级生物信息学多智能体系统
    
    特点：
    - 每个智能体都具有完整的工具调用能力
    - 通过AutoGen GroupChat实现多智能体协调
    - 支持异步和同步执行
    - 完整的错误处理和日志记录
    - 持久化结果存储
    """

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """        
        Args:
            llm_config: LLM配置
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
        self.llm_config = llm_config or self._setup_llm_config()
        
        # 工具映射 - 所有智能体共享的工具库
        self.tools = create_function_map()
        
        # 初始化智能体
        self._init_agents()

        # 将工具注册到各智能体，确保运行时可被LLM调用并执行
        try:
            self.register_tools_to_agents()
        except Exception:
            logger.warning("注册工具到智能体时发生错误，继续但请检查注册逻辑", exc_info=True)
        
        # 任务管理
        self.task_history = []
        self.current_task = None
        
        logger.info("✓ BioInfoMAS 生产级系统初始化完成")

    def _setup_llm_config(self) -> Dict[str, Any]:
        """设置LLM配置"""
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_API_KEY environment variable not set. "
                "Please configure your LLM API key."
            )
        
        config = {
            "config_list": [
                {
                    "model": os.getenv("LLM_MODEL_ID", "gpt-4"),
                    "api_key": api_key,
                    "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                }
            ],
            "timeout": int(os.getenv("AUTOGEN_TIMEOUT", "300")),
            "cache_seed": int(os.getenv("AUTOGEN_CACHE_SEED", "40")),
            "temperature": 0.7,
        }
        
        logger.info(f"LLM Config: {config['config_list'][0]['model']}")
        return config

    def _init_agents(self):
        """初始化所有智能体，配置工具调用能力"""
        
        # 1. PlannerAgent - 协调者
        self.planner_agent = PlannerAgent(self.llm_config)
        planner_agent = self.planner_agent.get_agent()
        planner_agent.register_for_llm(
            description="规划智能体，负责任务分解、工作流编排和多智能体协调"
        )   
        

        # 2. DataProcessingAgent - 数据获取和预处理
        self.data_agent = DataProcessingAgent(self.llm_config)
        data_agent = self.data_agent.get_agent()
        data_agent.register_for_llm(
            description="数据处理智能体，根据数据概况，决定清洗策略和预处理方式"
        )

        # 3. FeatureScreeningAgent - 知识推理
        self.feascureen_agent = FeatureScreeningAgent(self.llm_config)
        feascureen_agent = self.feascureen_agent.get_agent()
        feascureen_agent.register_for_llm(
            description="特征筛选智能体，利用统计相关性和LLM的语义知识，快速剔除无关变量，缩小搜索空间"
        )
        
        # 4. CausalFeatureSelectionAgent - 数据分析
        self.causal_agent = CausalFeatureSelectionAgent(self.llm_config)
        causal_agent = self.causal_agent.get_agent()
        causal_agent.register_for_llm(
            description="因果特征选择智能体，负责寻找局部因果特征"
        )
        

        # 5. ValidationAgent - 可视化和报告
        self.visualization_agent = ValidationAgent(self.llm_config)
        visualization_agent = self.visualization_agent.get_agent()
        visualization_agent.register_for_llm(
            description="验证智能体，复制验证局部因果结构"
        )
        
        logger.info("✓ 5个专业化智能体已初始化")

    def register_tools_to_agents(self):
        """为所有智能体注册工具调用能力。
        只将对应工具和函数映射注册到各自智能体，避免把全部工具暴露给所有智能体。
        """

        try:
            self.data_agent.register_tools()
        except Exception:
            logger.warning("为 DataProcessingAgent 注册工具失败", exc_info=True)

        try:
            self.causal_agent.register_tools()
        except Exception:
            logger.warning("为 CausalFeatureSelectionAgent 注册工具失败", exc_info=True)

        try:
            self.feascureen_agent.register_tools()
        except Exception:
            logger.warning("为 FeatureScreeningAgent 注册工具失败", exc_info=True)

        try:
            self.visualization_agent.register_tools()
        except Exception:
            logger.warning("为 ValidationAgent 注册工具失败", exc_info=True)

        logger.info("✓ 已为各智能体按需注册工具")
        

    def run_analysis(
        self,
        research_goal: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行完整的分析工作流
        
        Args:
            research_goal: 研究目标
            parameters: 可选参数
        
        Returns:
            分析结果
        """
        
        logger.info(f"开始新任务: {research_goal}")
        
        self.current_task = {
            "id": len(self.task_history) + 1,
            "goal": research_goal,
            "parameters": parameters or {},
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "results": {}
        }
        
        try:
            # 步骤1: 协调者分析需求
            logger.info("[步骤1] 协调者分析需求...")
            workflow_plan = self._plan_workflow(research_goal)
            self.current_task["workflow_plan"] = workflow_plan
            
            # 步骤2: 执行多智能体工作流
            logger.info("[步骤2] 执行多智能体协作分析...")
            results = self._execute_multi_agent_workflow(
                research_goal,
                workflow_plan
            )
            
            self.current_task["results"] = results
            self.current_task["status"] = "completed"
            self.current_task["end_time"] = datetime.now().isoformat()
            
            logger.info("✓ 任务完成")
            
            # 保存任务
            self.task_history.append(self.current_task)
            
            return {
                "status": "success",
                "task_id": self.current_task["id"],
                "goal": research_goal,
                "results": results
            }
        
        except Exception as e:
            self.current_task["status"] = "error"
            self.current_task["error"] = str(e)
            self.current_task["end_time"] = datetime.now().isoformat()
            
            self.task_history.append(self.current_task)
            
            logger.error(f"❌ 任务错误: {str(e)}", exc_info=True)
            
            return {
                "status": "error",
                "task_id": self.current_task["id"],
                "goal": research_goal,
                "error": str(e)
            }

    def _plan_workflow(self, research_goal: str) -> Dict[str, Any]:
        """
        使用协调者智能体规划工作流 - 真实LLM调用
        """
        orchestrator = self.orchestrator.get_agent()
        
        # 构建规划提示
        planning_prompt = f"""
        用户的研究目标：{research_goal}
        
        请分析这个需求，并提出详细的分析工作流方案：
        1. 需要的数据来源和数据类型
        2. 具体的分析步骤
        3. 需要调用哪些工具和智能体（每个步骤可调用一个智能体）
        
        返回JSON格式的工作流计划，包含以下字段：
        - goal: 研究目标
        - stages: 阶段列表，每个阶段包含 stage, agent,  description
        """
        
        try:
            # 调用LLM进行真实的工作流规划
            logger.info("调用LLM进行工作流规划...")
            
            response = orchestrator.generate_reply(
                messages=[{"content": planning_prompt, "role": "user"}],
                sender=self.orchestrator.get_agent(),
            )
            
            logger.info(f"LLM规划响应: {response}")
            
            # 尝试从响应中解析JSON
            try:
                # 查找JSON部分
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    workflow_plan = json.loads(json_match.group())
                    logger.info(f"✓ 成功解析工作流计划: {len(workflow_plan.get('stages', []))} 个阶段")
                    return workflow_plan
            except json.JSONDecodeError:
                logger.warning("无法解析LLM返回的JSON，使用默认计划")
        
        except Exception as e:
            logger.warning(f"LLM调用失败 ({str(e)})，使用默认工作流计划")
        
        
        logger.info(f"工作流计划: {len(workflow_plan['stages'])} 个阶段")
        return workflow_plan

    def _execute_multi_agent_workflow(
        self,
        research_goal: str,
        workflow_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行真实的多智能体工作流
        """
        
        results = {}
        history = ""
        
        for stage in workflow_plan["stages"]:
            stage_name = stage["stage"]
            agent_name = stage["agent"]
            stage_description = stage.get("description", "")
            logger.info(f"执行阶段: {stage_name} (使用 {agent_name})")
            
            
            # 为该阶段构建任务
            stage_task = self._execute_stage(
                agent_name,
                stage_name,
                workflow_plan["stages"],
                research_goal,
                stage_description,
                history
            )
           
            history += f"阶段 {stage_name} 使用 {agent_name} \n结果: {stage_task}\n\n"
            results[stage_name] = stage_task
        
        return results

    def _execute_stage(
        self,
        agent_name: str,
        stage_name: str,
        workflow_plan: str,
        research_goal: str,
        stage_description: str = "",
        history: str = ""
    ) -> Dict[str, Any]:
        """
        执行单个分析阶段 - 真实LLM调用
        """
        
        # 根据agent名称获取对应的智能体
        agent_map = {
            "DataProcessingAgen": self.data_agent,
            "FeatureScreeningAgent": self.feascureen_agent,
            "CausalFeatureSelectionAgent": self.causal_agent,
            "ValidationAgent": self.visualization_agent,
        }
        
        agent_instance = agent_map.get(agent_name)
        if not agent_instance:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = agent_instance.get_agent()
        # print(agent.description)
        # 为该阶段构建任务提示
        task_prompt = self._build_stage_prompt(history, workflow_plan, research_goal, stage_description)
        
        logger.info(f"[LLM调用] 发送任务给 {agent_name}...")
        logger.debug(f"任务提示:\n{task_prompt}")
        
        # 执行阶段
        stage_result = {
            "agent": agent_name,
            "stage": stage_name,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Function-calling loop: LLM 可以在响应中返回 JSON 格式的 function_calls，
            # 形如: {"function_calls": [{"name": "download_data", "arguments": {...}}]}
            # 我们会解析这些调用、在本地执行对应函数（来自 self.tools），
            # 并把执行结果作为一条来自工具的消息回传给 LLM，直到 LLM 不再请求调用工具。
            # 只将当前阶段允许的工具作为 functions 传入模型
            # allowed_tools_defs = [t for t in self.tools_definition if t.get("name") in tools]
            # functions_schema = convert_tools_to_openai_functions(allowed_tools_defs)

            messages = [{"role": "user", "content": task_prompt}]

            final_response = None
            max_rounds = 3
            for round_idx in range(max_rounds):
                logger.info(f"[Agent 调用] 轮次 {round_idx+1} -> 使用 {agent_name}.generate_reply()")

                # 让 agent 决定是否进行函数调用（优先使用 agent 提供的接口）
                try:
                    print(messages)
                    resp = agent.generate_reply(
                        messages=messages,
                        sender=agent,
                    )
                except Exception as e:
                    logger.warning(f"agent.generate_reply 调用失败 ({e}), 回退到直接模型调用", exc_info=True)
                    # 回退：使用全局模型 function-calling
                    resp = None
                print("当前轮次使用",agent_name,"的响应：", resp)
                
                # 解析 agent 响应：可能为 dict（含 function_call）、也可能为字符串（可能包含 JSON 表示）
                func_call = None
                parsed_resp_text = None

                if isinstance(resp, dict):
                    # 直接来自 agent 的结构化响应
                    func_call = resp.get("function_call") or resp.get("function_calls")
                    parsed_resp_text = resp.get("content")
                else:
                    # 尝试把文本解析为 JSON，寻找 function_calls 或 function_call
                    parsed_resp_text = str(resp or "")
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', parsed_resp_text, re.DOTALL)
                        if json_match:
                            parsed_json = json.loads(json_match.group())
                            func_call = parsed_json.get("function_call") or parsed_json.get("function_calls")
                    except Exception:
                        func_call = None

                if func_call:
                    # 标准化 func_call 为单个调用（或列表）
                    calls = func_call if isinstance(func_call, list) else [func_call]
                    for call in calls:
                        fname = call.get("name")
                        raw_args = call.get("arguments") or call.get("args") or {}

                        try:
                            if isinstance(raw_args, str):
                                fargs = json.loads(raw_args) if raw_args.strip() else {}
                            else:
                                fargs = raw_args or {}
                        except Exception:
                            fargs = {}

                        logger.info(f"{agent_name} 请求函数调用: {fname} args={fargs}")

                        func = self.tools.get(fname)
                        if not func:
                            err_msg = f"工具未找到: {fname}"
                            logger.warning(err_msg)
                            messages.append({"role": "function", "name": fname, "content": json.dumps({"error": err_msg}, ensure_ascii=False)})
                            continue

                        try:
                            if isinstance(fargs, dict):
                                result = func(**fargs)
                            else:
                                result = func(fargs)
                        except Exception as e:
                            logger.error(f"执行工具 {fname} 时出错: {str(e)}", exc_info=True)
                            messages.append({"role": "function", "name": fname, "content": json.dumps({"error": str(e)}, ensure_ascii=False)})
                            continue

                        try:
                            func_content = json.dumps({"result": result}, ensure_ascii=False)
                        except Exception:
                            func_content = str(result)

                        # 把函数执行结果回传给 agent
                        messages.append({"role": "user", "content": fname +"已执行完毕。结果："+func_content})
                        # 继续下一轮循环，让 agent 基于工具结果决定后续动作
                        continue
                else:
                    # agent 未请求函数调用，或 agent 返回为最终文本
                    final_response = parsed_resp_text or resp
                    logger.info(f"[Agent 响应] {agent_name} 完成 {stage_name} (无更多函数调用请求)")
                    break

            if final_response is None:
                final_response = resp.get("content") or resp.get("raw")

            stage_result["status"] = "completed"
            stage_result["llm_output"] = final_response

            # 尝试解析结构化输出
            try:
                # 如果 final_response 是 dict（raw），把它处理为字符串然后解析
                if isinstance(final_response, dict):
                    parsed_final = final_response
                else:
                    parsed_final = self._parse_llm_output(final_response, stage_name)

                stage_result["output"] = parsed_final
            except Exception as e:
                logger.warning(f"无法解析LLM输出: {str(e)}")
                stage_result["output"] = {"raw_response": final_response}

        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            stage_result["status"] = "error"
            stage_result["error"] = str(e)
            stage_result["output"] = {}
        
        return stage_result

    def _build_stage_prompt(
        self,
        history: str,
        workflow_plan: Dict[str, Any],
        research_goal: str,
        stage_description: List[str]
    ) -> str:
        """构建阶段任务提示"""
        
        prompts = f"""用户研究目标: {research_goal}

# 完整计划:
{workflow_plan}

# 历史步骤与结果:
{history if history else "无"}

# 当前步骤:
{stage_description}

请仅输出针对"当前步骤"的回答:
"""
        
        instruction = (
            "\n\n请注意：返回必须为 JSON 格式。若需要调用本地工具，请在 JSON 中包含 `function_calls` 字段，"
            "它应为数组，数组元素为 {\"name\": 工具名, \"arguments\": {...}}。"
            "一旦工具被执行，系统会将工具执行结果回传给你，之后你可以基于结果继续生成最终的 `final_output` 字段。"
        )

        return prompts + instruction

    def _parse_llm_output(self, response: str, stage_name: str) -> Dict[str, Any]:
        """
        解析LLM输出为结构化数据
        """
        import re
        
        # 尝试从响应中提取JSON数据
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                logger.info(f"✓ 成功解析 {stage_name} 的LLM输出")
                return parsed
        except:
            pass
        
        # 如果没有JSON，返回原始响应
        return {
            "response": response,
            "type": stage_name
        }

    def save_results(self, output_dir: str = "./results") -> str:
        """
        保存分析结果
        
        Args:
            output_dir: 输出目录
        
        Returns:
            保存的文件路径
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存当前任务
        if self.current_task:
            task_file = os.path.join(output_dir, f"task_{self.current_task['id']}.json")
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_task, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ 任务结果已保存: {task_file}")
        
        # 保存历史记录
        if self.task_history:
            history_file = os.path.join(output_dir, "task_history.json")
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.task_history, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ 任务历史已保存: {history_file}")
        
        return output_dir

    def get_task_history(self) -> List[Dict[str, Any]]:
        """获取任务历史"""
        return self.task_history

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """获取当前任务"""
        return self.current_task

