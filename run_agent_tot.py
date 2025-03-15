import argparse
import json
import gym
import os
from util import set_random_seed
from llm_agent_tot import APIAgent_TOT, extract_paths
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.utils import DEBUG_PROD_SIZE

def main(mode, temperature, top_p, max_batch_size, max_gen_len):
    """运行 ToT Agent 交互 WebShop 环境"""
    # set_random_seed(0)

    print(f"Running with settings:")
    print(f"Mode: {mode}, Temperature: {temperature}, Top P: {top_p}")

    # 初始化 ToT Agent
    api_url = "XXXXXX"
    
    agent = APIAgent_TOT(api_url, temperature, top_p, max_gen_len, num_branches=2, depth=6)

    # 初始化 WebShop 环境
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)

    rewards = []
    all_episodes_results = []

    for episode in range(1):  
        obs = env.reset()
        print(f"\n[Episode {episode}] Initial Observation: {obs}")

        # 生成轨迹树
        trajectory_tree = agent.tot_process(env, obs)
        
        # 提取所有从根到叶的轨迹和最终 reward
        episode_paths = extract_paths(trajectory_tree)
        
        # 将每个 episode 的结果保存在一个字典中
        episode_result = {
            "episode": episode,
            "paths": episode_paths
        }
        all_episodes_results.append(episode_result)
        
        print(f"[Episode {episode}] Results collected.")


    env.close()

    # 保存所有 episode 的结果到 JSON 文件
    result_path = "tot_agent_results.json"
    dir_name = os.path.dirname(result_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(all_episodes_results, file, indent=4, ensure_ascii=False)

    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--max_gen_len", type=int, default=512)

    args = parser.parse_args()

    main(
        mode=args.mode,
        temperature=args.temperature,
        top_p=args.top_p,
        max_batch_size=args.max_batch_size,
        max_gen_len=args.max_gen_len,
    )
