import streamlit as st
#from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import time

#import speech_recognition as sr



# The vectorstore we'll be using
from langchain.vectorstores import FAISS

# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA

# The easy document loader for text
from langchain.document_loaders import TextLoader

# The embedding engine that will convert our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter


#def speech_to_text():
#    r = sr.Recognizer()
#    with sr.Microphone() as source:
#        audio = r.listen(source)
#        try:
#            text = r.recognize_google(audio)
#            return text
#        except:
#            text = 'Ohhh, I can\'t hear your dj voice. Please sing to me again!'



openai_api_key=st.secrets["open_ai"]

import os
os.environ['OPENAI_API_KEY'] = st.secrets["open_ai"]






SysMessage = SystemMessage(content='''
You are an empathetic assistant, working for Adore Me and 
helping female customers figure out their bras, panties and lingeri sizes. 
You will use the document that we'll be providing for questions and answers and whatever you can't find there you don't know.
When you don't know something, you will say that you don't know how to answer that question Your responses will not be longer than
200 characters.
''')
loader = TextLoader('brasizing01.txt')

#st.title('Size Helper')
st.image("AM_logo2.png",  width = 310)
st.write("Hi, I am Celeste, your virtual assistant to help with bra size and fit. ")
# or choose one of the precanned questions

option = st.selectbox(
    label='The most frequently asked questions!',
    options=('What are the steps to measure my bra size at home?',
    'What do I do when I experience gaping?',
    'What do I do when I experience spillage?',
    'What do I do if I am wearing my straps too tight?',
    'Give me some pro tips to find my perfect size',
    'Give me a resource where I get all of this explained'),
    index=None)

    





doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)

# Get your embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
docsearch = FAISS.from_documents(docs, embeddings)



#make chat general
chat = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key)
qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever())
# give initial context
chat([SysMessage])

def generate_response(input_text):
    return qa.run(input_text)
    #return chat([HumanMessage(content=input_text)]).content

if "messages" not in st.session_state:
    st.session_state['messages'] = []
st.session_state['status']=''

if 'running' not in st.session_state:
    st.session_state['running'] = True

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if('button_message' not in st.session_state):
    st.session_state['button_message']="Talk to me baby!"

# voice part


if option:
    st.session_state.messages.append({"role": "user", "content": option})
    with st.chat_message("user"):
        st.markdown(option)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
    #    qa.run(input_text)
    all_response = generate_response(option)


    for response in all_response:
        full_response+=response
        time.sleep(0.003)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role":"assistant", "content":full_response})


st.divider()


if prompt := st.chat_input("Let's chat about your bra size!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
    #    qa.run(input_text)
    all_response = generate_response(prompt)

    for response in all_response:
        full_response+=response
        time.sleep(0.003)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role":"assistant", "content":full_response})

