import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_prompt

if __name__ == "__main__":
    # define model & tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # define article & question, target sentence: "[April2] CISA ..."
    with open("long_knowledge.txt", "r") as file:
        article = file.read().replace('\\n', '').replace('. ', '.')
    question = "What measures has CISA implemented or recommended to mitigate the risks associated with the Ivanti system vulnerabilities, and what steps should organizations take to prevent similar intrusions?"
    
    # split article into sentences using nltk
    sentences = sent_tokenize(article)
    
    # get prompt and apply template
    prompt = f"""\
Below is an article, read the article and answer my question after the article.
Now the article begins:
{article}
Now the article ends.
Select several sentences from the article to answer my question.
Question: {question}
"""
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

    eos_token_id = tokenizer.eos_token_id
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
