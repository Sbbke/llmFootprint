import preprocess.read_pdf as read_pdf
import pandas as pd
import tiktoken

def caculate_tokens(texts):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    tokens = encoding.encode(texts)
    return tokens


def ProcessedPdf(doc_titles, text_tokenization)->pd.DataFrame:
    
    contents = []
    headings = []
    tokens_per_section = []

    for title in doc_titles:
        path = f'{title}.pdf'
        num = 0
        texts = read_pdf.extract_text_from_pdf(path)
        # print(type(texts))
        
        # print(type(texts))
        processed_text = text_tokenization(texts)

        for i in processed_text:
            # print(i, "end of sentence")
            num = num + 1
            headings.append(f"{title} section {num}" )
            tokens = caculate_tokens(texts)
            tokens_per_section.append(len(tokens))
            contents.append(i)
        
    data = {
        'title' : headings,
        'content': contents,
        'tokens' : tokens_per_section
    }

    df = pd.DataFrame(data)
    df.columns = ['title', 'content', 'tokens']

    return df
    # print(df)
    # df.to_csv('test2.csv', index=False)
