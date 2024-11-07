
import pandas as pd
from embedding.embedding_openai import Embedding_openai
import np
import prompt_generate.similarity as similarity_caculator
import prompt_generate.prompt_generator as prompt_generator
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETIONS_MODEL = "gpt-4o-mini"

df  = pd.read_csv('test_calander.csv')
df.set_index('title')
embedding_model = Embedding_openai()

query = '我要預約'
query_embedding = []
with open('temp.txt', 'r') as f:
    for line in f:
        data = float(line.rstrip())
        # print(data)
        query_embedding.append(data)

# selected_section = prompt_generator.construct_prompt(query, vectorspace, df)

document_embeddings = embedding_model.compute_doc_embedding(df)
vectorspace = pd.read_csv('vector_space.csv')

document_similarities = sorted([
    (similarity_caculator.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in document_embeddings.items()
], reverse=True)
chosen_sections = []
chosen_sections_indexes = []
# print(document_similarities)
# for i in document_similarities:
    # print(type(i), i)

test = []
# for doc in document_similarities:
#     if doc[0]<0.4:
#         test.append(doc)
# print(type(document_similarities))
# print(df.keys())
for _, section_index in document_similarities:
    # Add contexts until we run out of space.        
    document_section = df.loc[section_index]
    chosen_sections.append('\n' + document_section.content.replace("\n", " "))
    chosen_sections_indexes.append(str(section_index))

# print(type(df))
prompt_generator.construct_prompt(query,vectorspace,df)
# print(type(vectorspace))
# print(similarity_caculator.order_by_similarity(query,vectorspace))
# for doc_index, doc_imbedding in document_embeddings.items():
#     print(doc_index, doc_imbedding)
# print(document_similarities)

