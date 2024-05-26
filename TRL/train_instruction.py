from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl.trainer import SFTTrainer
from utils import ConstantLengthDataset

# 加载数据集
raw_datasets = load_dataset("/root/CY/java")

# 初始化分词器
old_tokenizer = AutoTokenizer.from_pretrained("/root/CY/gpt2")

# 加载预训练的GPT-2模型
model = AutoModelForCausalLM.from_pretrained("/root/CY/gpt2").cuda()

# 创建训练和验证数据集
def prepare_dataset(example):
    # 将func_name和docstring作为输入的一部分，并作为prompt
    prompt = f"Function Name: {example['func_name']}\nDocstring: {example['docstring']}\n"
    # 将原始代码追加到prompt后面
    code = example['original_string']
    # 合并prompt和代码，形成完整的输入文本
    full_text = prompt + code
    return {"full_text": full_text}

# 加载数据集并应用过滤器
train_data = raw_datasets["train"].map(prepare_dataset)
eval_data = raw_datasets["validation"].map(prepare_dataset)

# 截断超长序列的函数
def truncate_long_sequences(example, max_length):
    full_text = example['full_text']
    # 截断超长序列
    truncated_text = full_text[:max_length]
    return {"full_text": truncated_text}

# 应用截断函数到数据集中的每个样本
train_data = train_data.map(lambda x: truncate_long_sequences(x, max_length=1000), batched=True)
eval_data = eval_data.map(lambda x: truncate_long_sequences(x, max_length=1000), batched=True)

# 创建数据集
train_dataset = ConstantLengthDataset(
    old_tokenizer,  # 使用原始的分词器
    train_data,
    formatting_func=lambda x: x['full_text'],  # 使用full_text作为样本文本
    infinite=True,
    seq_length=1000
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="/root/CY/CodeSearchNet/gpt2_java",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
)

# 初始化trl训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_data,
    formatting_func=lambda x: x['full_text'],  # 使用full_text作为样本文本
    max_seq_length=512
)

try:
    # 开始训练
    trainer.train()
    print("Training completed successfully.")
except Exception as e:
    print(f"Training failed with an error: {e}")

# 保存模型和分词器
try:
    model.save_pretrained("/root/CY/CodeSearchNet/gpt2_java")
    print("Model has been saved successfully.")
except Exception as e:
    print(f"Failed to save the model: {e}")

try:
    old_tokenizer.save_pretrained("/root/CY/CodeSearchNet/gpt2_java")
    print("Tokenizer has been saved successfully.")
except Exception as e:
    print(f"Failed to save the tokenizer: {e}")