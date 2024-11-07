# from langchain.text_splitter import CharacterTextSplitter
import langchain.text_splitter as text_splitter
class Preprocess_langchain():
    # def text_tokenization(self,texts:str):
    #     text_spiltter = text_splitter.CharacterTextSplitter(
    #         separator= '\n',
    #         chunk_size=100,
    #         chunk_overlap=20,
    #         length_function=len,
    #         is_separator_regex=False,
    #     )
    #     return text_spiltter.split_text(texts)

    def text_tokenization(self, texts:str):
        text_spiltter = text_splitter.RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        return text_spiltter.split_text(texts)


