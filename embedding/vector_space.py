# from embedding_openai import Embedding_openai
import pandas as df

def createVectorSpace(pd, embedding_model) -> df.DataFrame:
    # df  = pd.read_csv('test2.csv')
    # df.set_index('title')
    # embedding_model = Embedding_openai()
    doc_embedding = embedding_model.compute_doc_embedding(pd)

    # print(type(doc_embedding))
    # print(doc_embedding)
    vectorspace = df.DataFrame(doc_embedding)
    return vectorspace
    # vectorspace.to_csv('vector_space_section.csv', index=False)