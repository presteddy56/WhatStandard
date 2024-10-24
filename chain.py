import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains.llm import LLMChain #StuffDocumentsChain

class process:
    def template(self):
        template = (""" You are a medical doctor and advisor who precisely answers questions related to standard treatments in the diagnosis.
                Use the provided guidelines to answer the questions. I should use as many documents provided as possible.
                And also, you should mention the sentences exactly.
                If you don't know the answer, say so. 
                The guideline was provided from various countries. Therefore, you should support the users to understand the difference within the countries in terms of medical direction. 
                Then, provide a reference guideline with page numbers to support reviewing the details afterwards.
                It is helpful for you that the guidelines apply a format to indicate the page number with guideline names as header and footer. 
                If you add something outside the given guideline, you should mention that references are from outside of the given guidelines.

                Guidelines:{guidelines}
                
                Question:{question}
                
                Answer:""")
        document_prompt = PromptTemplate(
                    input_variables=["page_content"], template="{page_content}"
                    )
        self.document_prompt = document_prompt
        self.document_variable_name = "guidelines"
        prompt = PromptTemplate(
            template=template, input_variables=["guidelines", "question"]
            )
        self.prompt = prompt
    def setting_retriever(self, retriever):
        self.retriever = retriever
    def llm(self, temperature:float=0.1):
        groq_api_key = os.environ.get("GROQ_API_KEY")
        llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=groq_api_key, temperature = 0.1)
        self.llm = llm
    def chaining(self):
        chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=self.prompt,
                document_prompt=self.document_prompt,
                document_variable_name=self.document_variable_name,
                )
        self.chain = chain
    def run(self,query,doc):
        self.answer = self.chain.invoke(input= {"guidelines":doc, "question":query})
    
