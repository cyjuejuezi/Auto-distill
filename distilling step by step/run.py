# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse  #解析命令行参数，使用户可以自定义脚本的执行

from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer   #用于从预训练模型加载分词器
# 包含了不同数据集的加载器，用于加载指定的数据集
from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
# 提供了不同的评估指标计算函数
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
# 用于执行模型的训练和评估过程
from train_utils import train_and_evaluate


def run(args):# args参数包含了所有通过命令行传入的参数。
#准备数据集: 根据命令行参数指定的数据集名称，使用相应的加载器加载数据集。
#如果通过命令行指定--dataset cqa，则使用CQADatasetLoader()加载器

    #### Prepare datasets

    if args.dataset == 'cqa':
        dataset_loader = CQADatasetLoader()
    elif args.dataset == 'svamp':
        dataset_loader = SVAMPDatasetLoader()
    elif args.dataset == 'esnli':
        dataset_loader = ESNLIDatasetLoader()
    elif args.dataset == 'anli1':
        dataset_loader = ANLI1DatasetLoader()
#由于asdiv用于扩充SVAMP数据集，这里创建了两个加载器：一个用于加载SVAMP数据集，另一个用于加载ASDiv数据集。
    elif args.dataset == 'asdiv':  # NOTE: for augmenting SVAMP only
        dataset_loader = SVAMPDatasetLoader()
        dataset_loader_svamp = SVAMPDatasetLoader()
        dataset_loader_asdiv = ASDivDatasetLoader()
    else:
        raise ValueError

#如果是asdiv数据集，这段代码首先分别加载SVAMP和ASDiv的数据集，然后使用concatenate_datasets
# 函数将它们的训练集合并，而测试集则直接使用SVAMP的测试集。这种方式允许模型在一个更丰富的训练集上进行训练
    if args.dataset == 'asdiv':
        datasets_svamp = dataset_loader_svamp.load_from_json()
        datasets_asdiv = dataset_loader_asdiv.load_from_json()
        datasets = DatasetDict({
            'train': concatenate_datasets([datasets_svamp['train'], datasets_asdiv['train']]),
            'test': datasets_svamp['test']
        })
    else:
#对于除asdiv外的其他数据集，直接调用相应加载器的load_from_json方法来加载数据集。
# 这意味着每个数据集加载器都需要实现这个方法，返回一个包含训练集和测试集（有时还包括验证集）的数据集对象。
        datasets = dataset_loader.load_from_json() # 加载和处理数据集


#判断是否需要加载由大型语言模型（如palm或gpt）生成的预测结果,如果args.llm为None则不执行任何操作。
    if args.llm is None:
        pass
    elif args.llm == 'palm':
        if args.dataset == 'asdiv':
            # training set = SVAMP training + ASDiv training
#这部分代码专门针对asdiv数据集，并结合SVAMP数据集。它从两个
#数据集加载训练数据的LLM预测结果，然后将这些预测结果合并，形成最终的训练集和测试集使用的LLM预测。
            train_llm_rationales_svamp, train_llm_labels_svamp = dataset_loader_svamp.load_llm_preds(split='train')
            train_llm_rationales_asdiv, train_llm_labels_asdiv = dataset_loader_asdiv.load_llm_preds(split='train')
            train_llm_rationales = train_llm_rationales_svamp + train_llm_rationales_asdiv
            train_llm_labels = train_llm_labels_svamp + train_llm_labels_asdiv
            # test set = SVAMP test
            test_llm_rationales, test_llm_labels = dataset_loader_svamp.load_llm_preds(split='test')
        #从指定的数据集加载器中加载大型语言模型（LLM）为训练集和测试集生成的预测结果。这些预测结果可能包括模型预测的标签（train_llm
        #_labels和test_llm_labels）和/或模型生成的解释或理由（train_llm_rationales和test_llm_rationales）
        #dataset_loader.load_llm_preds(split='train')是调用数据加载器的load_llm_preds方法，分别针对训练集和测
        # 试集。这个方法的split参数指定了是加载训练集（'train'）还是测试集（'test'）的LLM预测结果。
        #返回值：load_llm_preds方法返回两个值：

#train_llm_rationales和test_llm_rationales：这些变量接收了对于训
#练集和测试集的LLM生成的理由或解释。这些理由可能是模型为其预测提供的文本解
#释，有助于理解模型的决策过程。
#train_llm_labels和test_llm_labels：这些变量接收了LLM为训练集和测试集预
#测的标签。这些预测标签可能用于直接比较、作为模型输入的一部分，或在某种形式的模型
# 训练中使用。

        else:
            train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
    elif args.llm == 'gpt':
        train_llm_rationales, train_llm_labels = dataset_loader.load_gpt_preds(split='train')
        test_llm_rationales, test_llm_labels = dataset_loader.load_gpt_preds(split='test')
    else:
        raise ValueError
#一旦加载了LLM的预测结果，这部分代码将这些预测结果作为新列添加到训练集和
# 测试集中。这允许在训练和评估模型时使用LLM的预测结果作为额外的信息或特征。
    if args.llm is not None:
        datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
        datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
        datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)


#对训练集进行下采样 :如果通过命令行参数指定了args.subsample且值小于1.0，这表示用户希望对训练集进行下采样，
# 以减少训练数据的数量。这部分代码使用train_test_split方法从原始训练集中分割出一部分数据
# 作为新的训练集，test_size参数指定了保留作为“测试”部分（实际上不使用）的数据比例，seed参数确保分割的可重复性。
    if args.subsample < 1.0:
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']
#代码检查dataset_loader是否有一个属性或方法has_valid来判断是否存在预先定义的验证集。
# 如果存在，则根据LLM类型（如palm或gpt），从加载器中加载验证集的LLM预测结果。如果不存在预先定义的验证集，则通过将训练集的10%分割出来创建验证集。
    if dataset_loader.has_valid:
        if args.llm is None:
            pass
        #针对palm或gpt指定的LLM，加载验证集的预测标签(valid_llm_labels)
        #和预测理由(valid_llm_rationales)
        elif args.llm == 'palm':
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(split='valid')
        elif args.llm == 'gpt':
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_gpt_preds(split='valid')
        else:
            raise ValueError

#代码向验证集添加了两个新列：LLM预测的标签(llm_label)和预测理由(llm_rationale)。
        datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
        datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
    else:
#如果不存在预定义的验证集，代码将执行分割操作并更新datasets对象，以包含新的训练集、验证集和测试集。
        train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

        datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
            'test': datasets['test'],
        })
#根据命令行参数args.label_type，如果指定使用LLM预测的标签（'llm'），则替换数据集中的原始标签，
#并使用LLM预测的标签。接着，代码计算在训练集和测试集上使用这些预测标签的准确率，并打印结果。
    if args.label_type == 'gt':
        pass
    elif args.label_type == 'llm' and args.llm is not None:
        if args.dataset not in ['svamp', 'asdiv']:
            train_label_acc = compute_text_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_text_acc(datasets['test']['llm_label'], datasets['test']['label'])
        else:
            train_label_acc = compute_equation_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_equation_acc(datasets['test']['llm_label'], datasets['test']['label'])

        print(f'LLM Train Acc: {train_label_acc:.4f}')
        print(f'LLM Test Acc: {test_label_acc:.4f}')

        datasets['train'] = datasets['train'].remove_columns('label')
        datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])

    else:
        raise ValueError
#使用了LLM，代码将移除数据集中原有的rationale列（如果存在），并将llm_rationale列重命名为rationale，使得所有后续操作都将使用由LLM生成的理由
    if args.llm is not None:
        if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
        datasets = datasets.rename_column('llm_rationale', 'rationale')
#添加部分
    if 'valid' not in datasets:
      train_valid_split = datasets['train'].train_test_split(test_size=0.2, seed=42)
      datasets = DatasetDict({
        'train': train_valid_split['train'],
        'valid': train_valid_split['test'],
        'test': datasets['test']
    })
    #### Prepare datasets Prepare data for training

#AutoTokenizer.from_pretrained方法根据args.from_pretrained指定的模型名（如'google/t5-v1_1-base'）自动选择并加载对应的分词器。
# 分词器用于将文本输入转换成模型能够处理的格式，包括将文本分解成令牌（token）、添加必要的特殊令牌（如序列起始、结束令牌）等。
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
#如果数据集是NLI相关的（即args.dataset包含字符串'nli'），这段代码将假设和前提连接成单个输入字符串。datasets.map应用一个函数到数据集的每
# 个样本上，这里使用的函数将每个样本的premise和hypothesis字段通过分词器的结束符（eos_token）连接起来，并移除原有的premise和hypothesis列。
    if 'nli' in args.dataset:
        datasets = datasets.map(
            lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
            remove_columns=['premise', 'hypothesis'],
        )

#这个分词函数为需要任务前缀的模型设计。它为输入文本添加了'predict: '或'explain: '前缀，分别用于主任务和解释性任务。
# 然后，使用分词器将文本转换为模型可处理的格式，包括输入ID和注意力掩码。同时，它还处理标签和理由的编码。
    if args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

            return model_inputs
#这个分词函数为标准模型准备。它直接对输入文本进行分词，不添加任何任务前缀，并处理标签的编码。
    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs

    else:
        raise ValueError

#这行代码将前面定义的tokenize_function应用到数据集的每个样本上。通过设置batched=True，允许函数以批量方式处理数据，提高效率。
    if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
    else:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            batched=True
        )

#这里根据数据集是处理文本任务还是方程任务，选择不同的评估函数。这些函数用于在模型训练和评估过程中计算指标，如准确率或方程求解的准确率。
    if args.model_type == 'standard':
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text_aux(tokenizer)
        else:
            compute_metrics = compute_metrics_equation_aux(tokenizer)

    else:
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text(tokenizer)
        else:
            compute_metrics = compute_metrics_equation(tokenizer)

#使用准备好的分词后的数据集和评估指标函数执行训练和评估。它接受命令行参数args、运行标识args.run、分词器tokenizer、
# 已分词的数据集tokenized_datasets和评估指标函数compute_metrics作为输入。
    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  #创建了一个ArgumentParser对象，它将用于处理命令行参数
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=3600)
    parser.add_argument('--eval_steps', type=int, default=125)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
#--dataset: 必需参数，指定要使用的数据集。
#--subsample: 可选参数，指定训练数据的子采样比例，默认为1.0（即不进行子采样）。
#--alpha: 可选参数，用于指定某些模型或训练策略中的超参数，默认值为0.5。
#--max_steps: 可选参数，训练的最大步数，默认为10000步。
#--eval_steps: 可选参数，指定每多少步进行一次评估，默认为250步。
#--batch_size: 可选参数，批量大小，默认为64。
#--optimizer_name: 可选参数，优化器名称，默认为'AdamW'。
#--lr: 可选参数，学习率，默认为5e-5。
#--from_pretrained: 可选参数，预训练模型的路径或标识符，默认为'google/t5-v1_1-base'。
#--label_type: 可选参数，标签类型，比如使用真实标签（'gt'）还是LLM预测的标签，默认为'gt'。
#--llm: 可选参数，指定是否使用大型语言模型（如'palm'）的预测，默认为'palm'。
#--model_type: 可选参数，模型类型，比如标准模型还是需要任务前缀的模型，默认为'task_prefix'。
#其他参数，如--bf16，--no_log，--output_rationale等，用于控制训练的特定方面或功能。


    args = parser.parse_args()

    run(args)
