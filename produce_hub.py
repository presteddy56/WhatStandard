import pdf_documentation
import contextual_retrieval
import vectores_langchain



def produce(self,query:str, data, k_1:float, b_:float):
        contextual_lists=contextual_retrieval.process()
        contextual_lists.chunk_lists = data
        contextual_lists.BM_25(query = query,k_1=k_1, b_=b_)
        contextual_lists.somantic()
        contextual_lists.WCscoring()
        contextual_lists.finally_selecting()
        vectores = vectores_langchain.process()
        vectores.clear()
        vectores.making_vectores(contextual_lists.contextual_retrieved)
        retriever = vectores.vectorstores.as_retriever()
        query = query
        return retriever, query