from langchain.text_splitter import CharacterTextSplitter

def text_tokenization(texts):
    text_spiltter = CharacterTextSplitter(
        separator= '\n',
        chunk_size = 10000,
        chunk_overlap = 200,
        length_function = len,
    )
    return text_spiltter.split_text(texts)