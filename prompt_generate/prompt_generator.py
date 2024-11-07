import sys
sys.path.append('../')
import prompt_generate.similarity
import pandas as pd
from openai import OpenAI
import os
import np
import tiktoken

MAX_SECTION_LEN = 50000
SEPARATOR = "\n* "
ENCODING = "gpt-4o-mini"  # encoding for text-davinci-003
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETIONS_MODEL = "gpt-4o-mini"

os.environ['OPENAIAPIKEY'] = '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'
client = OpenAI(api_key=os.getenv('OPENAIAPIKEY'))

encoding = tiktoken.encoding_for_model('gpt-4o-mini')
separator_len = len(encoding.encode(SEPARATOR))

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 2000,
    "model" : COMPLETIONS_MODEL
}

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame):
    """
    Fetch relevant 
    """
    most_relevant_document_sections = [] 
    relevant_document_sections = prompt_generate.similarity.order_by_similarity(question, context_embeddings)
    # print(type(relevant_document_sections))
    for doc in relevant_document_sections:
        # print(type(doc))
        if doc[0]>0.55:
            most_relevant_document_sections.append(doc)

    chosen_sections = []
    chosen_sections_indexes = []
    chosen_sections_len = 0
    # print(df.keys())
    # print(df.tokens)
    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[int(section_index)]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            print("out of tokens limit")
            break            

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
    print('Total tokens:', chosen_sections_len)
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
        
    return chosen_sections

def answer_with_gpt_4(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    messages = [
        {"role" : "system", "content":"你是一個真人客服, 依據提供的資訊與行事曆，以跟朋友聊天的方式提供操作流程的引導. 如果僅依據提供的資訊無法進行回答，就以十分抱歉的語氣說'我不知道'"}
    ]
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)

    context= ""
    for doc in prompt:
        context = context + doc 

    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})
    response = client.chat.completions.create(
        model=COMPLETIONS_MODEL,
        messages=messages
        )

    return '\n' + response.choices[0].message.content