import sys
sys.path.append('../')
from embedding.embedding_openai import Embedding_openai
import np

embedding_model = Embedding_openai()

def vector_similarity(x: list[float], y: list[float]) -> float:

    return np.dot(np.array(x), np.array(y))

def order_by_similarity(query: str, contexts: dict[(str, str), np.array]):

    query_embedding = embedding_model.get_embedding(query)
    # print(type(query_embedding))
    # with open('temp.txt', 'w') as f:
    #     for d in query_embedding:
    #         f.write(f"{d}\n")

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    print(document_similarities)
    return document_similarities

