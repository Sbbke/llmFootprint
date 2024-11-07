from openai import OpenAI
import os
import pandas as pd
import similarity
import unit_test.create_embedding as create_embedding
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETIONS_MODEL = "gpt-4o-mini"
os.environ['OPENAIAPIKEY'] = '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'
client = OpenAI(api_key=os.getenv('OPENAIAPIKEY'))

import np
df  = pd.read_csv('test_calander.csv')
df.set_index('title')
document_embeddings = create_embedding.compute_doc_embeddings(df)

# print(similarity.order_by_similarity("我想要預約健檢的話該怎麼做?", document_embeddings)[:5])

MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame):
    """
    Fetch relevant 
    """
    most_relevant_document_sections = [] 
    relevant_document_sections = similarity.order_by_similarity(question, context_embeddings)
    for doc in relevant_document_sections:
        if doc[0]>0.4:
            most_relevant_document_sections.append(doc)
    

    chosen_sections = []
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
        
    return chosen_sections


COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 2000,
    "model" : COMPLETIONS_MODEL
}

def answer_with_gpt_4(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    messages = [
        {"role" : "system", "content":"你是一個真人客服, 依據提供的資訊與行事曆回答，提供操作流程的教學. 若要預約健檢，依據提供的行事曆作回答，行事曆上有活動的日期不可以預約健檢. 如果僅依據提供的資訊無法進行回答，就以十分抱歉的語氣說'我不知道'"}
    ]
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)

    context= ""
    for article in prompt:
        context = context + article 

    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})
    response = client.chat.completions.create(
        model=COMPLETIONS_MODEL,
        messages=messages
        )

    return '\n' + response.choices[0].message.content
prompt = "那我要改預約11/12"
response = answer_with_gpt_4(prompt, df, document_embeddings)
print(response)