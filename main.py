import nltk
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_probs_list(input_ids, sentence_ids_list):
    prefix_map = {}
    for i, sentence_ids in enumerate(sentence_ids_list):
        prefix = sentence_ids[0][0].item()
        if prefix not in prefix_map:
            prefix_map[prefix] = []
        prefix_map[prefix].append(i)
    prefixes = list(prefix_map.keys())
    
    if len(prefix_map) == 1:
        probs_list = [[1] for _ in sentence_ids_list]    
    else:
        probs_list = [None for _ in sentence_ids_list]  
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        
        probs = torch.softmax(outputs.logits[0, -1, prefixes], dim=-1)
        for prefix, prob in zip(prefixes, probs):
            for idx in prefix_map[prefix]:
                probs_list[idx] = [prob.item()]
    
    for prefix, indices in prefix_map.items():
        if len(indices) == 1:
            continue
        
        next_input_ids = torch.cat([input_ids, torch.tensor(prefix).reshape(1, 1).cuda()], dim=-1)
        next_sentence_ids_list = [sentence_ids_list[idx][:, 1:] for idx in indices]
        next_probs_list = calculate_probs_list(next_input_ids, next_sentence_ids_list)
        
        for idx, probs in zip(indices, next_probs_list):
            probs_list[idx].extend(probs)
    
    return probs_list

def constrained_sentence_prefix_decoding(input_ids, sentence_ids_list, top_k=2):
    probs_list = calculate_probs_list(input_ids, sentence_ids_list)
    scores = [np.mean(np.log(probs)) for probs in probs_list]
    indices = np.argsort(scores)[::-1][:top_k]
    return [sentence_ids_list[idx] for idx in indices]

def find_subarray_index(a, b):
    n, m = len(a), len(b)
    for i in range(n - m + 1):
        if a[i : i + m] == b:
            return i
    return -1

def skip_decoding(input_ids, sentence_ids_list, max_length=256):
    intervals = []
    for sentence_ids in sentence_ids_list:
        sentence = tokenizer.decode(sentence_ids[0], skip_special_token=False)
        idx = find_subarray_index(article, sentence)
        sub_article_ids = tokenizer(article[idx:], return_tensors="pt", max_length=max_length, truncation=True).input_ids.cuda()
        
        sentence_input_ids = torch.cat([input_ids, sub_article_ids], dim=-1)
        with torch.no_grad():
            outputs = model(sentence_input_ids, use_cache=True)
        
        probs = torch.softmax(outputs.logits[0, -len(sub_article_ids[0]):, :], dim=-1)
        eos_idx = torch.argmax(probs[:, tokenizer.eos_token_id])
        evidence_sentence = tokenizer.decode(sub_article_ids[0, :eos_idx], skip_special_token=False)        
        intervals.append((idx, idx + len(evidence_sentence)))
    return intervals

def merge_interval(intervals):
    if not intervals or len(intervals) == 1:
        return intervals
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)

    return merged

if __name__ == "__main__":
    # define model & tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda().half()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # define article & question, target sentence: "[April 3] Researchers ..."
    with open("long_knowledge.txt", "r") as file:
        article = file.read().replace('\\n', '').replace('. ', '.')
    question = "What new attack methods used by the APT41 subgroup Earth Freybug were discovered, and how do they target endpoint protection systems?"
    
    # split article into sentences using nltk
    nltk.download('punkt_tab')
    sentences = nltk.tokenize.sent_tokenize(article)

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
    
    # apply Constrained Sentence Prefix Decoding & Skip Decoding & merge
    sentence_ids_list = [tokenizer(sentence, return_tensors="pt").input_ids.cuda() for sentence in sentences]
    sentence_ids_list = constrained_sentence_prefix_decoding(input_ids, sentence_ids_list, top_k=2)
    intervals = skip_decoding(input_ids, sentence_ids_list, max_length=256)
    intervals = merge_interval(intervals)    

    # show result
    for i, (start, end) in enumerate(intervals):
        print(f"------- high-quality knowledge {i + 1} -------")
        print(article[start : end])
