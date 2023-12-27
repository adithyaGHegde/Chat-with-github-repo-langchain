# for streamlit hosting
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers, HuggingFaceHub
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from htmlTemplate import css, bot_template, user_template
import git
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline

def get_text(docs):
    text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
            text += "\n\n"
    st.write(text)
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks  = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    general_system_template = r""" 
    You are a chat bot that helps the user by answering questions about the context provided using the documents. You read code and documentation
    as well as text files and help the user with their queries. Given a specific context, please give an answer to the question. If you don't 
    know the answer, please say "I don't know". 
    ----
    {context}
    ----
    """
    general_user_template = "Question:```{question}```"     # make better prompt templates!
    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )


    st.write("Creating conversation chain...")
    # llm = CTransformers(model='models/orca-mini-3b.ggmlv3.q4_0.bin', # Path to your model location
    #                 model_type='llama', 
    #                 config={'max_new_tokens': 1024,
    #                         'temperature': 0.01,
    #                         'context_length': 4000})
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",model_kwargs={'temperature':0.1,'max-length':1024})
    # llm = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True)
    
    # model_name = "Intel/dynamic_tinybert"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
    # question_answerer = pipeline(
    #     "question-answering", 
    #     model=model_name, 
    #     tokenizer=tokenizer,
    #     return_tensors='pt'
    # )
    # llm = HuggingFacePipeline(
    #     pipeline=question_answerer,
    #     model_kwargs={"temperature": 0.7, "max_length": 512},
    # )
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 6}),
        memory=memory,
        verbose = True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)

def clone_repo(gh_link):
    last_name = gh_link.split('/')[-1]
    clone_path = last_name.split('.')[0]
    if not os.path.exists(clone_path):
            # Clone the repository
            git.Repo.clone_from(gh_link, clone_path)
    return clone_path

allowed_extensions = ['.py', '.ipynb', '.md', '.cpp','.dart']
def get_docs(clone_path):
    root_dir = clone_path
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            file_extension = os.path.splitext(file)[1]
            if file_extension in allowed_extensions:
                try: 
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    docs.extend(loader.load_and_split())
                except Exception as e: 
                    pass
    return docs

def get_chunks_from_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    return texts

def delete_directory(clone_path):
    if os.path.exists(clone_path):
            for root, dirs, files in os.walk(clone_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(clone_path)

def get_vectorstore_from_docs(chunks,clone_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    delete_directory(clone_path)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with repo", page_icon=":books:", layout="wide")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.write(css, unsafe_allow_html=True)

    st.header("Chat with your data user!")
    user_question = st.text_input("What do you want to know about the repo/documents?")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:

        st.subheader("Talk to a Github repo")
        gh_link = st.text_input("Enter the github repo link")
        if gh_link:
            st.write("You entered:", gh_link)
            if st.button("Process repo"):
                with st.spinner("Processing..."):
                    clone_path = clone_repo(gh_link)
                    st.write("Your repo has been cloned")
                    docs = get_docs(clone_path)

                    # chunk the texts and store it in a vectorstore
                    chunks = get_chunks_from_docs(docs)
                    vectorstore = get_vectorstore_from_docs(chunks,clone_path)
                    st.write("Your repo has been processed.")

                    # create a conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

        st.write("OR")
        st.subheader("Talk to documents")
        docs = st.file_uploader("Upload a document to process", accept_multiple_files=True)
        if st.button("Process docs"):
            with st.spinner("Processing..."):
                # extract the text from the documents
                raw_text = get_text(docs)

                # chunk the texts and store it in a vectorstore
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                # create a conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
        

if __name__ == "__main__":
    main()
