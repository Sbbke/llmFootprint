import preprocess.read_pdf as read_pdf
import pandas as pd
import tiktoken
# nltk.download('stopwords')


# encoding = tiktoken.get_encoding("cl100k_base")


def caculate_tokens(texts):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    tokens = encoding.encode(texts)
    return tokens


def ProcessedPdf(doc_titles, text_tokenization)->pd.DataFrame:
    contents = []
    tokens_per_section = []
    
    for title in doc_titles:
        path = f'{title}.pdf'
        texts = read_pdf.extract_text_from_pdf(path)
        # print(type(texts))
        
        tokens = caculate_tokens(texts)
        tokens_per_section.append(len(tokens))

        # print(type(texts))
        processed_text = text_tokenization(texts)
        # print(type(processed_text))
        # processed_text2 = preprocessor1.text_tokenization_recursive(processed_text)
        # for i in processed_text2:
        #     print(i, "end of sentence")
        contents.append(processed_text)
        
    print(tokens_per_section)
        
    data = {
        'title' : doc_titles,
        'content': contents,
        'tokens' : tokens_per_section
    }

    df = pd.DataFrame(data)
    df. columns = ['title', 'content', 'tokens']
    return(df)



