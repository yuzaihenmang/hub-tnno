import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "./models/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "./output"
MAX_SEQ_LENGTH = 512


def create_chinese_dataset():
    data = [
        {
            "instruction": "请介绍一下人工智能。",
            "input": "",
            "output": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、感知和自然语言处理等能力。"
        },
        {
            "instruction": "什么是深度学习？",
            "input": "",
            "output": "深度学习是机器学习的一个子领域，使用具有多个层（深度）的神经网络来学习数据的复杂模式和表示。它模仿人脑的神经网络结构。"
        },
        {
            "instruction": "Python中的列表和元组有什么区别？",
            "input": "",
            "output": "列表（list）是可变的，可以修改、添加或删除元素，使用方括号[]。元组（tuple）是不可变的，一旦创建就不能修改，使用圆括号()。"
        },
        {
            "instruction": "解释一下什么是监督学习。",
            "input": "",
            "output": "监督学习是机器学习的一种方法，使用标记的训练数据来训练模型。模型学习输入和输出之间的映射关系，然后可以对新的未标记数据进行预测。"
        },
        {
            "instruction": "如何提高模型的泛化能力？",
            "input": "",
            "output": "提高模型泛化能力的方法包括：1) 增加训练数据量和多样性 2) 使用正则化技术（如Dropout、L2正则化）3) 数据增强 4) 交叉验证 5) 防止过拟合。"
        }
    ]

    formatted_data = []
    for item in data:
        text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        formatted_data.append({"text": text})

    return formatted_data


def create_causal_mask(input_ids, attention_mask, tokenizer):
    batch_size, seq_length = input_ids.shape

    # 创建基础下三角因果掩码
    causal_mask = torch.tril(torch.ones((seq_length, seq_length)))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    # 创建分段掩码（基于<|im_end|>分隔符）
    segment_mask = torch.ones((batch_size, seq_length, seq_length))

    for i in range(batch_size):
        tokens = input_ids[i]
        sep_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        if sep_token_id is not None:
            sep_positions = torch.where(tokens == sep_token_id)[0]
            if len(sep_positions) >= 1:
                # 第一个segment结束位置
                first_segment_end = sep_positions[0]
                # 第二个segment开始位置
                second_segment_start = first_segment_end + 1

                if second_segment_start < seq_length:
                    # 第一个segment只看到自己
                    segment_mask[i, :first_segment_end + 1, :first_segment_end + 1] = 1
                    segment_mask[i, :first_segment_end + 1, second_segment_start:] = 0

                    # 第二个segment可以看到第一个segment
                    if len(sep_positions) >= 2:
                        second_segment_end = sep_positions[1]
                        segment_mask[i, second_segment_start:second_segment_end + 1, :second_segment_end + 1] = 1

    segment_mask = segment_mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]

    # 合并掩码
    combined_mask = causal_mask * segment_mask

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        combined_mask = combined_mask * attention_mask

    return combined_mask


def patch_attention_for_causal_mask():
    import transformers
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    original_forward = Qwen2Attention.forward

    def new_forward(self, hidden_states, attention_mask=None, **kwargs):
        if attention_mask is not None and attention_mask.dim() == 2:
            batch_size, seq_length = hidden_states.shape[:2]
            device = hidden_states.device

            # 创建基础因果掩码
            causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=device))
            causal_mask = causal_mask.view(1, 1, seq_length, seq_length)

            # 如果有attention_mask，合并
            if attention_mask is not None:
                mask_2d = attention_mask
                mask_4d = mask_2d[:, None, None, :]
                causal_mask = causal_mask * mask_4d

            attention_mask = causal_mask

        return original_forward(self, hidden_states=hidden_states, attention_mask=attention_mask, **kwargs)

    Qwen2Attention.forward = new_forward
    return original_forward


def load_model_and_tokenizer():
    print(f"加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 应用注意力掩码补丁
    patch_attention_for_causal_mask()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def preprocess_function(examples, tokenizer):
    # Tokenize文本
    result = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

    # 将输入转换为tensor用于创建掩码
    input_ids = torch.tensor(result["input_ids"])
    attention_mask = torch.tensor(result["attention_mask"])

    # 创建因果+分段掩码
    causal_attention_mask = create_causal_mask(input_ids, attention_mask, tokenizer)

    # 将掩码转换回列表格式
    batch_size, _, seq_length, _ = causal_attention_mask.shape
    causal_attention_mask = causal_attention_mask[:, 0, :, :]  # 取第一个head

    # 创建有效的attention_mask（批处理中每个序列的掩码）
    final_attention_mask = []
    for i in range(batch_size):
        # 取每一行的第一个token的掩码（因为因果掩码是对称的）
        mask_row = causal_attention_mask[i, 0, :].tolist()
        final_attention_mask.append(mask_row)

    result["attention_mask"] = final_attention_mask

    return result


def main():
    print("开始SFT微调训练（带因果掩码）")

    model, tokenizer = load_model_and_tokenizer()

    print("构造训练数据...")
    train_data = create_chinese_dataset()
    dataset = Dataset.from_list(train_data)

    # 预处理数据集，添加因果掩码
    print("预处理数据，创建因果掩码...")
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        batch_size=len(dataset)
    )

    print(f"训练样本数量: {len(dataset)}")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        report_to=None,
        dataloader_pin_memory=False,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("开始训练...")
    trainer.train()

    print(f"训练完成，保存模型到 {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("训练完成！")


if __name__ == "__main__":
    main()