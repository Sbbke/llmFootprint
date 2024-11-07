from openai import OpenAI
import os
import pandas as pd
EMBEDDING_MODEL = "text-embedding-3-small"
os.environ['OPENAIAPIKEY'] = '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'
client = OpenAI(api_key=os.getenv('OPENAIAPIKEY'))

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    response = client.embeddings.create(
      model=model,
      input=text
    )
    return response.data[0].embedding

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

