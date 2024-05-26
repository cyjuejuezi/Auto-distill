# hotpot.py

from vllm import LLM, SamplingParams
import wikienv
import torch

# 初始化模型
model = LLM(model="/root/CY/llama2-13b-chat")
print("Model initialized successfully.")

# 定义生成文本的函数
def llm(prompt, stop):
    try:
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=100,
            stop=stop
        )
        generated = model.generate(prompt, sampling_params)
        # 确保返回的是单个字符串，而不是列表或元组中的元素
        return generated[0].outputs[0].text if generated else ""
    except Exception as e:
        print(f"Error generating text: {e}")
        return ""

# 初始化环境
env = wikienv.WikiEnv()

# 定义step函数
def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            obs, reward, done, info = env.step(action)
            return obs, reward, done, info
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempts}, retrying...")
            attempts += 1
            time.sleep(1)  # 简单的退避策略
    print("Max attempts reached, failing.")
    raise requests.exceptions.Timeout("Max retries exceeded.")

# 执行示例
try:
    obs, info = env.reset()
    question = "What movie did actress Irene Jacob complete before the American action crime thriller film directed by Stuart Bird?"
    print(f"Question: {question}\n")

    # 假设我们已经有了一个初始的 observation
    # 这里我们需要一个循环来处理 Thought, Action, Observation 步骤
    for i in range(1, 3):  # 假设我们只需要两个循环来获取答案
        thought_action = llm(obs + f"Thought {i}: ", stop="\nAction")
        action = thought_action.strip().split('\n')[-1]  # 获取最后一个元素作为 action
        if action.lower().startswith("search"):
            # 从 action 中提取实体名称
            entity = action[action.find('[') + 1:action.find(']')]
            env.search_step(entity)  # 使用 WikiEnv 的 search_step 方法
            obs = env._get_obs()  # 更新观测值
        elif action.lower().startswith("lookup"):
            # 处理查找操作
            keyword = action[action.find('[') + 1:action.find(']')]
            # 这里需要实现查找逻辑，可能需要修改 WikiEnv 类
            pass
        else:
            print("Unsupported action.")
            break

    # 假设我们已经得到了答案
    # 我们需要调用 env.step("finish[answer]") 来结束这一轮对话
    obs, _, done, info = env.step("finish[answer]")

    print(info)
except Exception as e:
    print(f"Error: {e}")

