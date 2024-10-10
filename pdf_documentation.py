import os
import json
import PyPDF2 as pdf_loader

class process:
    def set_data(self,root_folder, source_folder, lists_file):
        self.root = root_folder
        self.source = source_folder
        with open(f"{self.root}/{self.source}/{lists_file}", 'r') as file:
            self.data = json.load(file)
        print(self.data)
    def documents(self,documents_folder,max_pages=40,tag_naming:str = "countries", data_naming:str ="files"):
        full_documents = {}
        for i in range(len(self.data[data_naming])):
            country_name = self.data[tag_naming][i]
            file_path = self.root+"/"+self.source+"/"+documents_folder+"/"+self.data[data_naming][i]
            contents = pdf_loader.PdfReader(file_path)
            pages = ""
            for m in range(max_pages):
                try:
                    page = contents.pages[m].extract_text()
                    pages=pages + page
                except:pass
            full_documents[country_name] = pages
        self.full_documents=full_documents
