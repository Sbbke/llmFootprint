from openai import OpenAI
import os


EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETIONS_MODEL = "gpt-4o-mini"
os.environ['OPENAIAPIKEY'] = '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'
client = OpenAI(api_key=os.getenv('OPENAIAPIKEY'))

prompt = "我想要預約健檢的話該怎麼做?"
message = [
    {"role" : "user", "content":prompt}
    ]
response = client.chat.completions.create(
    model= COMPLETIONS_MODEL,
    messages= message,
)

print(response.choices[0].message.content)