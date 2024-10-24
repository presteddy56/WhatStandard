import os
import json
import PyPDF2 as pdf_loader
from uuid import uuid4
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class process:
    def set_data(self,root_folder, source_folder, lists_file):
        self.root = root_folder
        self.source = source_folder
        with open(f"{self.root}/{self.source}/{lists_file}", 'r') as file:
            self.data = json.load(file)
    def documents(self,documents_folder,max_pages=40):
        full_documents = {}
        for i in range(len(self.data['files'])):
            country_name = self.data["countries"][i]
            file_path = self.root+"/"+self.source+"/"+documents_folder+"/"+self.data["files"][i]
            contents = pdf_loader.PdfReader(file_path)
            pages = ""
            for m in range(max_pages):
                try:
                    page = contents.pages[m].extract_text()
                    pages=pages + f"{country_name}_Page{m}-----<"+ page + f">-------{country_name}_Page{m}"
                except:pass
            full_documents[country_name] = pages
        self.full_documents=full_documents
    def vectorstore(self, store_folder):
        documents=[]
        for i in range(len(self.data["countries"])):
            document_i = Document(
                page_content= self.full_documents[self.data["countries"][i]],
                metadata={"source":self.data["countries"][i]},
                id=i,
            )
            documents.append(document_i)

    # Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(model_name = "mixedbread-ai/mxbai-embed-large-v1")
        self.vectorstore = Chroma(
            collection_name="WhatStandards",
            embedding_function=embeddings,
            persist_directory=f"{self.root}/{self.source}/chroma_langchain_db",  # Where to save data locally, remove if not necessary
        )

        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorstore.add_documents(documents=documents, ids=uuids)
    def retriever(self):
        self.retriever = self.vectorstore.as_retriever()
