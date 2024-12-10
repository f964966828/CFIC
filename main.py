from nltk.tokenize import sent_tokenize
from utils import get_prompt

with open("long_knowledge.txt", "r") as file:
    article = file.read().replace('\\n', '').replace('. ', '.')

# target sentence: "[April2] CISA ..."
question = "What measures has CISA implemented or recommended to mitigate the risks associated with the Ivanti system vulnerabilities, and what steps should organizations take to prevent similar intrusions?"

sentences = sent_tokenize(article)
prompt = get_prompt(article, question)
print(prompt)
