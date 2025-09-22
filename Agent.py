from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader       # python -m pip install langchain-community --use-pep517
from langchain_chroma import Chroma
from langchain.text_splitter import SpacyTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import shutil
import os

from dotenv import load_dotenv


load_dotenv()



persist_directory = 'db'

embedding_model = GoogleGenerativeAIEmbeddings(model ="models/embedding-001", google_api_key = os.getenv('GOOGLE_API_KEY'))

if not os.path.exists(persist_directory):
   # shutil.rmtree(persist_directory)
   loader = TextLoader("Stock.txt")
   data = loader.load()
   text = data[0].page_content

   splitter = SpacyTextSplitter(chunk_size = 200, chunk_overlap= 50)
   sentences = splitter.split_text(text)

   
   embeddings = embedding_model.embed_documents(sentences)

   chunks = []
   current_chunk = [sentences[0]]

   for i in range (1,len(sentences)):
       similarity = cosine_similarity([embeddings[i-1]],[embeddings[i]])[0][0]

       if similarity < 0.8 :
           chunks.append(" ".join(current_chunk))
           current_chunk = []

       current_chunk.append(sentences[i])

   chunks.append(" ".join(current_chunk))

    #print(chunks)    


   



   docs = [Document(page_content=chunk) for chunk in chunks]
   vector_db = Chroma.from_documents(
              documents = docs,
              embedding = embedding_model,
              persist_directory = persist_directory
              )
else:
    
    vector_db = Chroma(
        embedding_function = embedding_model,
        persist_directory = persist_directory
    )    


retriever = vector_db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(
    model = 'gemini-1.5-flash',
    google_api_key = os.getenv('GOOGLE_API_KEY')
)

prompt = PromptTemplate(
    input_variables =['context','question'],
     template =  """
           Answer the question only according to the following context:
           {context}

           Question : {question}              
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = ({"context":retriever|format_docs, "question":RunnablePassthrough()}
         |prompt
         |llm
         |StrOutputParser()
         )
#ret_chain = retriever|format_docs
#ret = ret_chain.invoke("What is a Stock Market Crash?")
try:
    output = chain.invoke("What is a stock Market Crash")

    print(output)

    print("-----------------------------------------------------------------------------------------------")
 
except Exception as e:
     print("❌ An error occurred while running the chain:")
     import traceback
     traceback.print_exc()

#print(ret)


