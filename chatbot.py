from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA 
from langchain.callbacks import StdOutCallbackHandler
from argparse import ArgumentParser
import json
import os
import chainlit as cl

with open('args.json', 'r') as f:
    args = json.loads(f.read())

model_path = args['model_path']
data_path = args['data_dir']
# Load Model
llm = LlamaCpp(
    model_path= model_path,
    n_gpu_layers=12,
    n_batch=256,
    use_mlock=True,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)   

# split text:
loader = PyPDFDirectoryLoader(data_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
texts = text_splitter.split_documents(docs)

# Get specific embeddings for model
embeddings = LlamaCppEmbeddings(
    model_path=model_path,  
    verbose=False,
)

# Create db if it doesnt exist

if os.path.isdir("./chroma_db"):
    
    # check if there are changes to filedirectory
    with open('dat_dir.json', 'r') as f:
        dir = json.loads(f.read())
        
    # remake chromadb
    if dir != os.listdir(data_path):
        print("Creating Chroma Database from documents... This may take a while.")
        with open('dat_dir.json', 'w') as f:
            json.dump(os.listdir(data_path), f)
        db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")
        db.persist()

    # grab db
    else:
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        db.get() 
    

else:
    print("Creating Chroma Database from documents... This may take a while.")
    with open('dat_dir.json', 'w') as f:
        json.dump(os.listdir(data_path), f)
    db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")
    db.persist()

# Get prompt template for chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# init chain
conversation_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 10}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
        callbacks=[StdOutCallbackHandler()]
    )   

# chainlit async
@cl.on_chat_start
async def start():
    chain = conversation_chain
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, What would you like to know about the documents?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]

    await cl.Message(content=answer).send()