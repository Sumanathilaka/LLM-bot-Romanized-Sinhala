from llama_index import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.prompts.prompts import SimpleInputPrompt
import os, yaml, re
from openai import OpenAI, ChatCompletion
import streamlit as st
from htmlTemplates import css, bot_template, user_template
from llama_index.response.pprint_utils import pprint_response

documents=SimpleDirectoryReader("data").load_data()
len(documents)

with open('credentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

os.environ['OPENAI_API_KEY'] = credentials['OPENAI_API_KEY']

#text-embedding-ada-002 for Word Embedding


from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

def pipeline():
    index=VectorStoreIndex.from_documents(documents,show_progress=True)
    #query_engine=index.as_query_engine() 

    retriever=VectorIndexRetriever(index=index,similarity_top_k=4)
    postprocessor=SimilarityPostprocessor(similarity_cutoff=0.80)
    query_engine=RetrieverQueryEngine(retriever=retriever,
                                    node_postprocessors=[postprocessor])
    return query_engine


#user input handling and chat 
def handle_userinput(user_question,query_engine):
    response=query_engine.query(user_question)
               
           
    st.write(user_template.replace(
                "{{MSG}}", user_question), unsafe_allow_html=True)
                       
        
    st.write(bot_template.replace(
                "{{MSG}}", response.response), unsafe_allow_html=True)

#main method
def main():
    #query_engine=pipeline()
    st.set_page_config(page_title="Chat with BankBot:",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

     
    st.header("Chat with BankBot :robot_face:")
    user_question = st.text_input("Ask a question about your documents:")
    
    query_engine=pipeline()
    if st.button("Ask"):
        with st.spinner("Searching for the best answer"):
            handle_userinput(user_question,query_engine)

    
if __name__ == '__main__':
    main()

