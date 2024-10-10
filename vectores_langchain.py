from uuid import uuid4
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS

class process:
    def clear(self):
        try:
            self.vectorestore.delate(ids=uuids[:])
        except:
            pass
    def making_vectores(self,retrieval_documents):
        documents=[]
        for i in retrieval_documents["countries"].unique():
            selected_documents=""
            for pop in retrieval_documents["text"].loc[retrieval_documents["countries"]==i]:
                selected_documents +=pop
        
            document_i = Document(
                page_content= selected_documents,
                metadata={"source":i},
                id=i,
            )
            documents.append(document_i)
        self.documents = documents
# Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(model_name = "mixedbread-ai/mxbai-embed-large-v1")
        vectorstore = FAISS.from_documents(self.documents,embeddings)

        uuids = [str(uuid4()) for _ in range(len(self.documents))]

        vectorstore.add_documents(documents=self.documents, ids=uuids)
        self.vectorstores = vectorstore

