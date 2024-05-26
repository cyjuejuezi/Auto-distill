# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from datasets import load_dataset
#
# # 加载分词器和模型
# tokenizer = AutoTokenizer.from_pretrained("/root/CY/llama3-8b")
# model = AutoModelForCausalLM.from_pretrained("/root/CY/llama3-8b")
#
# # 创建一个文本生成pipeline
# text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
#
# # 加载数据集
# dataset = load_dataset("/root/CY/java", split='test')
#
# # 定义一个函数来生成文本
# def generate_responses(prompts, max_length=512):
#     return text_generator(prompts, max_length=max_length, num_return_sequences=1, truncation=True)
#
# # 定义一个函数来构建输入文本
# def build_prompts(func_names):
#     prompts = ["Please output the code of the Function without any annotation and sentence except code. Function: " + fn for fn in func_names]
#     return prompts
#
# # 选择几个测试样本
# test_samples = dataset.select(range(1))  # 选取测试集前5个样本
#
# # 构造输入
# prompts = build_prompts(test_samples['func_name'])
#
# # 生成回应
# results = generate_responses(prompts)
#
# # 打印输入和输出
# for idx, result in enumerate(results):
#     input_prompt = prompts[idx]  # 获取使用的具体提示
#     generated_text = result[0]['generated_text'] if result else "No output generated"
#     print(f"Input Prompt: {input_prompt}")
#     print(f"Generated Code:\n{generated_text}")
#     print("===")

##################################################################################################
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from datasets import load_dataset
# import numpy as np
# from nltk.translate.bleu_score import sentence_bleu
#
# # 加载分词器和模型
# tokenizer = AutoTokenizer.from_pretrained("/root/CY/llama3-8b")
# model = AutoModelForCausalLM.from_pretrained("/root/CY/llama3-8b")
#
# # 创建一个文本生成pipeline
# text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
#
# # 加载数据集
# dataset = load_dataset("/root/CY/java", split='test')  # 确保路径是正确的
#
# # 定义一个函数来生成文本
# def generate_responses(prompts, max_length=512):
#     return text_generator(prompts, max_length=max_length, num_return_sequences=1)
#
# # 定义一个函数来构建输入文本
# def build_prompts(func_names):
#     prompts = ["Please output the code of the Function. Function: " + fn for fn in func_names]
#     return prompts
#
# # 选择几个测试样本
# test_samples = dataset.select(range(2))  # 选取测试集前10个样本
#
# # 构造输入
# prompts = build_prompts(test_samples['func_name'])
#
# # 生成回应
# results = generate_responses(prompts)
#
# # 提取生成的文本
# generated_texts = [result[0]['generated_text'] for result in results]
#
# # 计算 BLEU 分数的函数
# def bleu_score(refs, hyps):
#     total_score = 0.0
#     for ref, hyp in zip(refs, hyps):
#         ref_tokens = ref.split()
#         hyp_tokens = hyp.split()
#         score = sentence_bleu([ref_tokens], hyp_tokens)
#         total_score += score
#     avg_score = total_score / len(refs)
#     return avg_score
#
# # 生成参考文本（这里需要调整为适当的参考文本）
# original_strings = test_samples['original_string']  # 假设我们将函数代码作为参考文本
#
# # 计算 BLEU 分数
# score = bleu_score(original_strings, generated_texts)
# print("BLEU Score:", score)

#############################################################################################
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("/root/CY/llama3-8b")
model = AutoModelForCausalLM.from_pretrained("/root/CY/llama3-8b")

# 创建一个文本生成pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 加载数据集
dataset = load_dataset("/root/CY/java", split='test')  # 确保路径是正确的

# 定义一个函数来生成文本
def generate_responses(prompts, max_length=512):
    return text_generator(prompts, max_length=max_length, num_return_sequences=1)

# 定义一个函数来构建输入文本
def build_prompts(func_names, docstrings):
    prompts = [ds + " Function: " + fn
                    for fn, ds in zip(func_names, docstrings)]
    return prompts

# 选择几个测试样本
test_samples = dataset.select(range(2))  # 选取测试集前5个样本，注意更正之前的注释

# 构造输入
prompts = build_prompts(test_samples['func_name'], test_samples['docstring'])

# 生成回应
results = generate_responses(prompts)

# 提取生成的文本
generated_texts = [result[0]['generated_text'] for result in results]

# 定义平滑函数
smoothie = SmoothingFunction()

# 计算 BLEU 分数的函数，现在包括平滑处理
def bleu_score(refs, hyps, smoothing_function):
    total_score = 0.0
    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        # 使用平滑函数计算BLEU分数
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing_function.method1)
        total_score += score
    avg_score = total_score / len(refs)
    return avg_score

original_strings = test_samples['original_string']  # 假设我们将函数代码作为参考文本

# 计算 BLEU 分数
score = bleu_score(original_strings, generated_texts, smoothie)
print("BLEU Score with smoothing:", score)
