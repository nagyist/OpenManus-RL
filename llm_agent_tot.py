import json
import re
import gym
import requests
import random
import os
from typing import Dict, List, Any
from util import refine_prompt
from prompt.prompt_tot import (
    prompt_system_thought, prompt_user_thought, 
    prompt_system_action, prompt_user_action,
    prompt_system_demo, prompt_user_demo 
)

def extract_json(text):
    """提取 `{}` 内的 JSON 数据"""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def clone_env(env):
    """
    克隆环境。优先使用 env.clone() 方法，如果不存在，则尝试通过 gym.make 重新构造环境，
    并复制状态（如果支持 get_state 和 set_state）。
    """
    if hasattr(env, 'clone'):
        return env.clone()
    try:
        new_env = gym.make(env.spec.id)
        if hasattr(env, 'get_state') and hasattr(new_env, 'set_state'):
            new_env.set_state(env.get_state())
        else:
            print("Warning: Environment does not support get_state/set_state; new environment is reinitialized.")
        return new_env
    except Exception as e:
        raise RuntimeError("Failed to clone environment: " + str(e))

class APIAgent_TOT:
    def __init__(self, api_url, temperature, top_p, max_gen_len, num_branches=3, depth=2):
        self.api_url = api_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        self.num_branches = num_branches  # 分支数量
        self.depth = depth  # ToT 深度
        self.max_retries = 5
        self.timeout = 600  # 10 分钟超时


    def call_model_api(self, prompt):
        """调用 LLM API 生成文本，同时返回 logp（如果有）"""
        payload = json.dumps({"prompt": prompt, "max_new_tokens": self.max_gen_len})
        headers = {'Content-Type': 'application/json'}
        
        for _ in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, data=payload, timeout=self.timeout)
                response_data = json.loads(response.text)

                if "completions" in response_data and response_data["completions"]:
                    completion = response_data["completions"][0]
                    text = completion.get("text", None)
                    logp = completion.get("logp", None)  # 解析 logp
                    return text, logp  # 返回文本和 logp
            except requests.exceptions.RequestException:
                pass
        return None, None  # 失败时返回 None


    def generate_full_trajectory(self, env, obs, thought_history, actions_history, reward_history, depth):
        """
        递归生成单个轨迹节点，每个节点保存一个步骤的数据（以字典保存）以及子节点列表。
        """
        # 1. 生成新思维（Thought）
        thought_prompt = refine_prompt(prompt_user_thought, browser_content=str(obs), past_actions=", ".join(actions_history))
        current_thought_raw, thought_logp = self.call_model_api(prompt_system_thought + thought_prompt)
        try:
            current_thought = json.loads(current_thought_raw)['thought']
        except (json.JSONDecodeError, KeyError):
            current_thought = "None"
        print("current_thought", current_thought)

        # 2. 生成新动作（Action）
        action_prompt = refine_prompt(prompt_user_action, browser_content=str(obs), past_actions=", ".join(actions_history), thought=current_thought)
        current_action_raw, action_logp = self.call_model_api(prompt_system_action + action_prompt)
        current_action_parsed = extract_json(current_action_raw)
        if current_action_parsed is not None:
            current_action = str(current_action_parsed.get('action', 'None')) + '[' + str(current_action_parsed.get('action_params', '')) + ']'
        else:
            current_action = "None"
        print("current_action", current_action)

        # 3. 克隆环境并执行当前动作
        env_copy = clone_env(env)
        # 若环境不支持状态复制，则需要 reset 后重放历史动作
        if not (hasattr(env, 'get_state') and hasattr(env, 'set_state')):
            env_copy.reset()
            for past_action in actions_history:
                env_copy.step(past_action)
        new_obs, reward, done, _ = env_copy.step(current_action)
        new_thought_history = thought_history + [current_thought]
        new_actions_history = actions_history + [current_action]
        new_reward_history = reward_history + [reward]
        cumulative_reward = sum(new_reward_history)

        # 4. 记录当前步骤的数据，使用字典便于保存 JSON
        step_data = {
            "observation": str(obs),
            "thought_history": thought_history,
            "actions_history": actions_history,
            "current_thought": current_thought,
            "current_thought_logp": thought_logp,
            "current_action": current_action,
            "current_action_logp": action_logp,
            "cumulative_reward": cumulative_reward
        }

        # 构造当前节点
        node = {"step": step_data, "children": []}

        # 若未结束且深度未到达底部，则递归生成子节点
        if not done and depth > 1:
            for _ in range(self.num_branches):
                child_node = self.generate_full_trajectory(env_copy, new_obs, new_thought_history, new_actions_history, new_reward_history, depth - 1)
                node["children"].append(child_node)
        return node

    def tot_process(self, env, obs):
        """
        从初始状态生成一个轨迹树，将各分支放在一个虚拟根节点下。
        """
        children = []
        for _ in range(self.num_branches):
            child_node = self.generate_full_trajectory(env, obs, [], [], [], self.depth)
            children.append(child_node)
        root = {"step": None, "children": children}
        return root

def extract_paths(tree, path=None):
    """
    递归提取从根节点到叶子节点的所有轨迹。
    每条轨迹为一系列步骤（字典列表），同时记录最后一步的累计 reward。
    返回一个列表，每个元素形如：
       {
         "trajectory": [step_data1, step_data2, ..., step_dataN],
         "final_reward": reward
       }
    """
    if path is None:
        path = []
    current_path = path.copy()
    if tree["step"] is not None:
        current_path.append(tree["step"])
    if not tree["children"]:
        final_reward = current_path[-1]["cumulative_reward"] if current_path else None
        return [{"trajectory": current_path, "final_reward": final_reward}]
    paths = []
    for child in tree["children"]:
        paths.extend(extract_paths(child, current_path))
    return paths


if __name__ == "__main__":
    env = gym.make("YourEnv-v0")  # 替换为你的环境
    obs = env.reset()
    agent = APIAgent_TOT(api_url="http://your-llm-api", temperature=0.7, top_p=0.9, max_gen_len=100, num_branches=3, depth=3)

    # 生成轨迹树
    trajectory_tree = agent.tot_process(env, obs)
    # 提取所有从根到叶的轨迹和最终 reward
    all_paths = extract_paths(trajectory_tree)

    # 保存结果到 JSON 文件
    result_path = "tot_agent_results.json"
    dir_name = os.path.dirname(result_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(all_paths, file, indent=4, ensure_ascii=False)

    print(f"Results saved to {result_path}")
