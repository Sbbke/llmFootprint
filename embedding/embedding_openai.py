from openai import OpenAI
import os
import pandas as pd

EMBEDDING_MODEL = "text-embedding-3-small"


class Embedding_openai():
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key=os.getenv('OPENAIAPIKEY'))

    def compute_doc_embedding(self,df):
        return {
            idx: self.get_embedding(r.content) for idx, r in df.iterrows()
        }
        
    def get_embedding(self, text: str, model: str=EMBEDDING_MODEL) -> list[float]:
        response = self.client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

