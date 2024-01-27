#https://www.peoplesbank.lk/user-manual/

#Queries
#labena samanya sewa mnwda
#naya sewa walata adala wisthara monawada
#sthira thanpathu walata adala thorathuru laba dnna
#ginum dekk athara mudal huwamaru karnne keseda

import streamlit as st
import os, yaml, textwrap, urljoin
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from htmlTemplates import css, bot_template, user_template
import TranslaterLogic, Transliterator
from deep_translator import GoogleTranslator

with open('credentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

os.environ['OPENAI_API_KEY'] = credentials['OPENAI_API_KEY']

chat_llm = ChatOpenAI(
                openai_api_key = os.environ['OPENAI_API_KEY'],
                model = 'gpt-3.5-turbo',
                temperature=0.5,
                max_tokens=500
                )
embedding_llm = HuggingFaceBgeEmbeddings(
                                        model_name = "BAAI/bge-small-en",
                                        model_kwargs = {'device': 'cuda'},
                                        encode_kwargs = {'normalize_embeddings': False}
                                        )


DATA_PATH='data/'

#cleaning the repository
def clean_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            # Iterate over all files in the directory and remove them
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Directory '{directory_path}' cleaned successfully.")
        else:
            print(f"Directory '{directory_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
#File scraping using hte BeautifulSoup
def scrape_pdfs(url):
    clean_directory(DATA_PATH)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    pdf_links = []
    for link in soup.find_all('a', href=True):
        if link['href'].endswith('.pdf'):
            if link['href'][0]=="/":
                pdf_links.append(url+link['href'])
            elif link['href'][:4]=="http":
                pdf_links.append(link['href'])
            else:
                pdf_links.append(url+"/"+link['href'])

            
    print(pdf_links)
    download_pdfs(pdf_links)


    
def download_pdfs(pdf_links):
    for link in pdf_links:
        filename = os.path.basename(link)
        response = requests.get(link, stream=True)
        print(response)

        if response.status_code == 200:
            with open(os.path.join(DATA_PATH, filename), 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
        else:
            print(f"Download failed for {filename}. Status code: {response.status_code}")
    
#Reading the Scraped pdf from the directory
def get_pdf_text(pdf_dir):
    pdf_docs = [os.path.join(pdf_dir, pdf) for pdf in os.listdir(pdf_dir)]
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#Creating the tex Chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
                                        separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        )
    chunks = text_splitter.split_text(text)
    return chunks

#buidling the vector store
def get_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(
                                    texts=text_chunks, 
                                    embedding=embedding_llm
                                    )
    return vectorstore

#Pipeline
def data_pipeline(url):
    scrape_pdfs(url)
    text = get_pdf_text(DATA_PATH)
    chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(chunks)
    return vectorstore 

#Conversation chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
                                    memory_key='chat_history', 
                                    return_messages=True
                                    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                                            llm=chat_llm,
                                                            retriever=vectorstore.as_retriever(),
                                                            memory=memory
                                                            )
    return conversation_chain

#Translation and Transliteration modueles
def translate_text_sinhala_to_english(text):
    translator = GoogleTranslator(source="si", target="en")
    translation = translator.translate(text)
    return translation

def translate_text_english_to_sinhala(text):
    translator = GoogleTranslator(source="en", target="si")
    translation = translator.translate(text)
    return translation

#user input handling and chat 
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    
    j=len(st.session_state.sinhalaTextLst)-1
    #print(j)
    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            msg=st.session_state.sinhalaTextLst[j]
            j=j-1
            #print(j)
            st.write(user_template.replace(
                "{{MSG}}", msg), unsafe_allow_html=True)
                       
        else:
            st.write(bot_template.replace(
                "{{MSG}}", translate_text_english_to_sinhala(message.content)), unsafe_allow_html=True)

#main method
        
def main():
    sinhalaTextLst=[]
    st.set_page_config(page_title="Chat with BankBot:",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "sinhalaTextLst" not in st.session_state:
        st.session_state.sinhalaTextLst = []    

    st.header("Chat with BankBot :robot_face:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if st.button("Ask"):
        with st.spinner("Searching for the best answer"):
            sinhalaText=Transliterator.triGramTranslate(user_question)
            st.session_state.sinhalaTextLst.append(sinhalaText)            
            english_text=translate_text_sinhala_to_english(sinhalaText)
            handle_userinput(english_text)

    with st.sidebar:
        st.subheader("Your document location")
        web_url = st.text_input("Enter the Website  URL")
        if st.button("Process"):
            with st.spinner("Processing"):
                # scrape the pdf text
                vectorstore=data_pipeline(web_url)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
       
        

if __name__ == '__main__':
    main()
