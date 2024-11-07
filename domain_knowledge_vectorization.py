import preprocess.data_preprocess_section as SectionPreprocessor
from preprocess.Tokenization.tokenization_langchain import Preprocess_langchain

import embedding.vector_space as vector_space
from embedding.embedding_openai import Embedding_openai

import pandas as pd

file_titles = ['flow_chart','calander_test','user_story']
tokenizer = Preprocess_langchain()
processed_doc = pd.DataFrame()
processed_doc = SectionPreprocessor.ProcessedPdf(file_titles,tokenizer.text_tokenization)
processed_doc.to_csv('temp.csv', index=False)
# print(processed_doc.index)
embedding_model = Embedding_openai()
doc_vector_space = pd.DataFrame()
doc_vector_space = vector_space.createVectorSpace(processed_doc, embedding_model)
doc_vector_space.to_csv('vs_temp.csv', index=False)


