import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()

eos_token_id = tokenizer.eos_token_id  # 句子結束標記 (GPT-2 沒有明確 [EOS]，通常使用長度限制)

# 開始生成
generated_ids = input_ids  # 初始化生成序列
for _ in range(50):
    outputs = model(generated_ids)
    logits = outputs.logits
    print(logits.shape)
    
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
    generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)

    if next_token_id.item() == eos_token_id:
        break
    
generated_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Sentence:", generated_sentence)

