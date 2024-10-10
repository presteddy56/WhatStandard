import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, StuffDocumentsChain

class process:
    def template(self):
        template = (""" You are a medical doctor and advisor who precisely answers questions related to standard treatment in a diagnosis.
                Use the provided guidelines to answer the questions. You should mention the sentences exactly.
                If you don't know the answer, say so. The guideline was provided from various countries.Therefore, you should make the differences clear as much as possible. 
                Then, provide a reference in the given guidelines to review the details. 
                You should refer to the given guidelines with pages. 
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
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_prompt=self.document_prompt,
                document_variable_name=self.document_variable_name,
                )
        self.chain = chain
    def run(self,query,doc):
        self.answer = self.chain.run(input_documents= doc, question = query)
    
