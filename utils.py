
def get_prompt(article, question):
    return f"""\
Below is an article, read the article and answer my question after the article.
Now the article begins:
{article}
Now the article ends.
Select several sentences from the article to answer my question.
Question: {question}
"""

