import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from vectorstore import get_vectorstore_from_url
from retriever_chain import get_context_retriever_chain


def get_conversational_rag_chain(retriever_chain,api_key): 
    llm = ChatOpenAI(openai_api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query,api_key):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store,api_key)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain,api_key)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']



# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with Knovatek")
openAiKey = st.text_input(label="Input the openai api key", type="password")



# sidebar
with st.sidebar:
    st.header("Spiderweb")
    website_url = st.text_input("Enter your URL")

if website_url is None or website_url == "":
    st.subheader("Please enter a website URL")

else:
    # session state
    try:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, Welcome to Knovatek. How can I help you?"),
            ]
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(website_url,openAiKey)    



        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            response = get_response(user_query,openAiKey)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
        
       

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

    except Exception as e:
            st.error(f"Error occurred: {str(e)}")