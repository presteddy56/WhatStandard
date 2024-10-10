import pdf_documentation
import contextual_retrieval
import vectores_langchain
import chain
pdfs =pdf_documentation.process()
pdfs.set_data("/Users/presteddy56","resources","doc_name.json")
pdfs.documents("pdfs")
crude_data =pdfs.full_documents
contextual_lists = contextual_retrieval.process()
contextual_lists.set_documents(crude_data)
contextual_lists.tokenized_splitter(512)
def chain_processing(query:str,k1:float=1.5, b:float=0.75, percentile:float=0.95,temperature=0.1):
    try:
        retrieval_sample = contextual_retrieval.process()
        retrieval_sample.chunk_lists= contextual_lists.chunk_lists
        retrieval_sample.BM_25(query=query, k_1=k1,b_=b)
        retrieval_sample.somantic()
        retrieval_sample.WCscoring()
        retrieval_sample.finally_selecting(percentile = percentile)
        shaped_sample=retrieval_sample.contextual_retrieved
        vectores_sample=vectores_langchain.process()
        vectores_sample.clear()
        vectores_sample.making_vectores(shaped_sample)
        retriever_sample=vectores_sample.vectorstores.as_retriever()
        doc_sample = vectores_sample.documents
        chain_sample = chain.process()
        chain_sample.template()
        chain_sample.setting_retriever(retriever_sample)
        chain_sample.llm(temperature = temperature)
        chain_sample.chaining()
        chain_sample.run(query, doc_sample)
        return chain_sample.answer
    except:
        alarm_comment ="Sorry, I tried to capture flooding data from guidline😤. I am happy if you change prompts or parameter for me.I will do my best👍"
        return alarm_comment
    
import gradio as gr

title = "Quick reference to guidelines for people with depression and clinicians"
description = """
<center>
<img src="https://github.com/presteddy56/WhatStandard/blob/main/images/image_1.png?raw=true" alt="logo" width="100"/>
</center>
"""

demo = gr.Interface(
    title = title,
    description = description,
    fn = chain_processing,
    inputs = "text",
    additional_inputs=[
        gr.Slider(value=1.2,minimum=0,maximum=3,step=0.1,label="k1: the impact of term frequency saturation. If you don't need to make the texts saturated, increase k1."),
        gr.Slider(value=0.75, minimum=0,maximum=1,step=0.1, label="b: document length normalization.When the document is more general and have more topics, increase b "),
        gr.Slider(value=0.95,minimum=0.75, maximum=1,step=0.001,label="Percentile: the amount of input data"),
        gr.Slider(value=0.1, minimum=0, maximum=1,step=0.1, label="temperature: productivity and stability of LLM")],
    outputs= "text",
    live = False,
    allow_flagging="never",
    theme= gr.themes.Monochrome()
)
demo.launch()