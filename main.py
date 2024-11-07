from openai import OpenAI
import os
import pandas as pd
from prompt_generate import prompt_generator

EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETIONS_MODEL = "gpt-4o-mini"

file_path = os.getcwd()+'/vector_space_by_document'
df  = pd.read_csv(file_path+'/sections.csv')
df.set_index('title')
vector_space = pd.read_csv(file_path+'/vector_space.csv')

prompt = "我要怎麼預約健檢"
response = prompt_generator.answer_with_gpt_4(prompt, df, vector_space)
print(response)