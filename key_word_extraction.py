import jieba
import jieba.analyse

def get_key_words(text):
    tags = jieba.analyse.extract_tags(text, topK=20, withWeight=True)
    for tag in tags:
        print('word:', tag[0], 'tf-idf:', tag[1])
    return tags

