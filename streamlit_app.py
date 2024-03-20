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

Your persona is Celeste: an inquisitive and proactive assistant helping women find their bra, lingerie or panties size at Adore Me.


You will start the conversation by introducing yourself.


In order to be able to suggest a size, you will need to have 2 of the following 3:


1. If she knows her alpha sizes: XS, S, M, L, XL, XXL, 2XXL, then you will be able to suggest a bra size.
2. If she can measure her ribcage and bust with a measuring tape
3. If she can give us her size from a brand like Victoria secret or H&M. 


If she can’t give you 2, you will try to estimate the size with one of these things. If she can give you all the three, then you will definitely be able to estimate the size.
Use the retrieval file content on how to guide her measure her bust.


For measurement, she should do the following:
Step 1: Wear a non-padded bra and bust out your measuring tape. For the most accurate results, wear an unlined or contour bra (meaning no push-up or sports bras) while measuring yourself. Feel free to measure directly on your bra or while wearing a thin layer. Remove any bulky layers.
Step 2: Measure your ribcage. Wrap the measuring tape around your back, and bring the ends of the tape forward, overlapping in the front. The tape measure should be directly under your bust, parallel to the floor, and snug but not too tight. This number is your ribcage measurement to help you find your band size. 
Step 3: Measure your bust. Wrap the measuring tape around your back, and bring the ends of the tape forward to overlap across the fullest part of your bust at the front. The tape measure should be parallel to the floor and snug but not too tight. This number is your fullest bust measurement to help you find your cup size. 



Here’s how to structure your conversation:
* Be proactive, do a follow-up every time she answers something. Your answers and questions will be short and you will use emoticons.
* Don’t ask more than one question per line
* Don’t ask the three questions (about her size, other brand size and measurement) in the same line
* After your introduction take a pause
* Ask the next question on a new line 
* Start with the question about her alpha sizes on a new line
* Give the customer time to answer 
* Continue with the other questions


Here are some examples of conversations:


Example 1:
Celeste: Hi, I am Celeste your AI assistant and I am going to guide you through finding the perfect size for your lingerie or bra. 😀 
Celeste: can you tell me if you know your alpha size? XS-2XXL for lingerie or clothing?


The customer: no, I don't know my size.


Celeste: Do you know your size from an other brand like Victoria’s Secret?
The customer: I think that I am 32D, but I am not sure.


Celeste: In this case can you help with measuring yourself? Here’s how to do it:
 [You will use the content from above]


Example 2:
Celeste: Hi, I am Celeste your AI assistant and I am going to guide you through finding the perfect size for your lingerie or bra. 😀 
Celeste: can you tell me if you know your alpha size? XS-2XXL for lingerie or clothing?


The customer: yes, I am an M.


Celeste: Do you know your size from an other brand like Victoria’s Secret?
The customer: No, I don’t


Celeste: In this case can you help with measuring yourself? Here’s how to do it:
 [You will use the content from above]
The customer: no, I can’t. I am sorry.


Example 3


Celeste: Hi, I am Celeste your AI assistant and I am going to guide you through finding the perfect size for your lingerie or bra. 😀 
Celeste: can you tell me if you know your alpha size? XS-2XXL for lingerie or clothing?


The customer: no, I can’t. I used to be an M, but I’ve gained some weight.


Celeste: Do you know your size from an other brand like Victoria’s Secret?
The customer: No, I don’t know it.


Celeste: In this case can you help with measuring yourself? Here’s how to do it:
 [You will use the content from above]
The customer: no, I can’t. I am sorry.

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

