import os
import requests
from transformers import AutoTokenizer
import json
import PyPDF2 as pdf_loader
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import scipy

class process:
    def set_documents(self, documents:dict):
        self.crude_documents = documents
    def tokenized_splitter(self, token_setting:int):
        self.token_setting = token_setting
        def split_into_many(text: str, tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased",clean_up_tokenization_spaces=False), max_tokens:int=self.token_setting) -> list:
            """ Function to split a string into many strings of a specified number of tokens """

    
            sentences = text.split('. ') #A
            n_tokens = []
            new_sentences = ""
            for sentence in sentences:
                TK = tokenizer(" " + sentence, padding=True, add_special_tokens=True,truncation=True, max_length=max_tokens)
                n_tokens.append(len(TK["input_ids"]))
                sentence += ". "
                new_sentences += sentence
    
            new_sentences = new_sentences.split(". ")
        #B
            chunks = []
            tokens_so_far = 0
            chunk = []

            for sentence, token in zip(new_sentences, n_tokens): #C

                if tokens_so_far + token > max_tokens: #D 
                    chunks.append(". ".join(chunk) + ".")
                    chunk = []
                    tokens_so_far = 0

                if token > max_tokens:#E 
                    continue
                chunk.append(sentence) #F
                tokens_so_far += token + 1

            return chunks
#A Split the text into sentences
#B Get the number of tokens for each sentence
#C Loop through the sentences and tokens joined together in a tuple
#D If the number of tokens so far plus the number of tokens in the current sentence is greater than the max number of tokens, then add the chunk to the list of chunks and reset
#E If the number of tokens in the current sentence is greater than the max number of tokens, go to the next sentence
#F # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        def tokenize(text,max_tokens:int=self.token_setting) -> pd.DataFrame:
            """ Function to split the text into chunks of a maximum number of tokens """

    
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",clean_up_tokenization_spaces=True) #A

            limitted_doc = split_into_many(text, tokenizer,max_tokens)
            df = pd.DataFrame({'title':[], 'text':[],'n_tokens':[]})
            for i in range(len(limitted_doc)):
                df_tmp= pd.DataFrame({'title':[], 'text':[],'n_tokens':[]})
                df_tmp.at[0,'title']=0
                texts=str(limitted_doc[i])
                df_tmp.at[0,'text'] = texts
                TK = tokenizer(" " + texts, padding=True, add_special_tokens=True,truncation=True, max_length=512)
                df_tmp['n_tokens']= len(TK["input_ids"])
                df = pd.concat([df, df_tmp], axis=0)
        #B
            shortened = []

            for row in df.iterrows():

                if row[1]['text'] is None: #C
                    continue

                if row[1]['n_tokens'] > max_tokens: #D
                    shortened += split_into_many(row[1]['text'], tokenizer, max_tokens)

                else: #E
                    shortened.append(row[1]['text'])

            df = pd.DataFrame(shortened, columns=['text'])
    
            df['n_tokens'] = df.text.apply(lambda x: len(tokenizer(x, padding=True, add_special_tokens=True,truncation=True, max_length=512)["input_ids"]))

            return df
#A Load the transformaer tokenizer which is designed to work with the distilbert-base-uncased
#B Tokenize the text and save the number of tokens to a new column
#C If the text is None, go to the next row
#D If the number of tokens is greater than the max number of tokens, split the text into chunks
#E Otherwise, add the text to the list of shortened texts
        tokenized = pd.DataFrame(({'text':[], 'n_tokens':[],'countries':[]}))
        for i in self.crude_documents:
            single_tokenized = tokenize(self.crude_documents[i],self.token_setting)
            single_tokenized['countries'] = i
            tokenized = pd.concat([tokenized,single_tokenized],axis=0, ignore_index=True)
        self.chunk_lists = tokenized

    def BM_25(self,query, k_1=1.5, b_=0.75):
        self.query = query
        full_corpus = list(self.chunk_lists["text"])
        def BM25_calculator(corpus:list, query:str):
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus,k1=k_1,b=b_)
            tokenized_query = query.split(" ")
            doc_scores=bm25.get_scores(tokenized_query)
            return doc_scores

        bm_data = BM25_calculator(full_corpus, self.query)
        self.chunk_lists["BM25"] = bm_data
        self.chunk_lists["BM25_rank"]=self.chunk_lists["BM25"].rank(ascending=False)
    def somantic(self):
        hugging_face_api_key = os.environ.get("HF_TOKEN")
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        hf_token = hugging_face_api_key
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        headers = {"Authorization": f"Presteddy56 {hf_token}"}
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        texts = list(self.chunk_lists["text"])
        output = embedding_model.encode(texts)
        self.chunk_lists["embeddings"] = list(output)
        query_embedding = embedding_model.encode(self.query)
        cos =[]
        for emb in self.chunk_lists["embeddings"]:
            cos_tmp = scipy.spatial.distance.cosine(emb,query_embedding)
            cos.append(cos_tmp)
        self.chunk_lists["cosine"]=cos
        self.chunk_lists["cosine_rank"]=self.chunk_lists["cosine"].rank(ascending=False)
    def WCscoring(self):
        self.chunk_lists["WCscore"]= (self.chunk_lists["BM25"]/self.chunk_lists["BM25_rank"])+(self.chunk_lists["cosine"]/self.chunk_lists["cosine_rank"])
    def finally_selecting(self, percentile:float=0.95):
        quantile_rate = percentile
        self.contextual_retrieved = self.chunk_lists.loc[self.chunk_lists["WCscore"] >= self.chunk_lists["WCscore"].quantile(quantile_rate)]

    

