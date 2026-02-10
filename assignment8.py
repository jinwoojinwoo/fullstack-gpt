import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from urllib.parse import urlparse
from pathlib import Path

Path("./.cache/embeddings").mkdir(parents=True, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        

def print_history():
    for msg in st.session_state["messages"]:
        send_message(
            msg["message"],
            msg["role"],
            save=False,
        )

def load_history():
    msgs = st.session_state.get("messages", [])
    history = []

    # exclude the last human message
    # include last 20 messages 
    for msg in msgs[-21:-1]:
        r = msg["role"]
        m = msg["message"]
        if r == "human":
            history.append(HumanMessage(content=m))
        elif r == "ai":
            history.append(AIMessage(content=m))
    return history

url = "https://developers.cloudflare.com/sitemap.xml"

answers_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
     Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                    
     Then, give a score to the answer between 0 and 5.

     If the answer answers the user question the score should be high, else it should be low.

     Make sure to always include the answer's score even if it's 0.

     Context: {context}
                                                    
     Examples:
                                                    
     Question: How far away is the moon?
     Answer: The moon is 384,400 km away.
     Score: 5
                                                    
     Question: How far away is the sun?
     Answer: I don't know
     Score: 0
     """),
     MessagesPlaceholder(variable_name="history"),
     ("human", "{question}"),

])

st.set_page_config(
    page_title="SiteGPT",
    page_icon="⛴️",
)

st.markdown(
    """
    # SiteGPT

    Ask questions about the content of Cloudflare's documentation.

    Start by writing OpenAI api_key on the sidebar.
    """
)

h2t = Html2TextTransformer()

def get_answers(inputs):
    api_key = inputs["api_key"]
    docs = inputs["docs"]
    question = inputs["question"]

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key=api_key,
    )

    answers_chain = answers_prompt | llm

    return {
        "api_key": api_key,
        "question": question, 
        "answers": [
            {
                "answer": answers_chain.invoke({
                        "question": question, 
                        "context": doc.page_content,
                        "history": load_history(),
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            } for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            If a source is available, return the source as-is. Do not include the score or the date in the response.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    api_key = inputs["api_key"]
    answers = inputs["answers"]
    question = inputs["question"]

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        openai_api_key=api_key,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" 
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

def parse_page(soup):
    header = soup.find("header")
    nav = soup.find("nav")
    footer = soup.find("div", class_="custom-footer-section")
    if header:
        header.decompose()
    if nav:
        nav.decompose()
    if footer:
        footer.decompose()

    return (
        str(soup.get_text())
        .replace('\n', ' ')
    )

@st.cache_data(show_spinner="Loading website...")
def load_website(url, api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[
            r'^(.*\/ai-gateway\/).*',
            r'^(.*\/vectorize\/).*',
            r'^(.*\/workers-ai\/).*',
        ],
        parsing_function=parse_page,
    )
    
    docs = loader.load_and_split(text_splitter=splitter)

    url_domain = urlparse(url).netloc
    cache_dir = LocalFileStore(f"./.cache/embeddings/{url_domain}")
    embeddings = OpenAIEmbeddings(
        chunk_size=100, 
        openai_api_key=api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()
    

with st.sidebar:
    api_key = st.text_input(
        "OpenAI API Key",
    )

    st.link_button("github", "https://github.com/jinwoojinwoo/fullstack-gpt")


  
if api_key:
    retriever = load_website(url, api_key)

    print_history()

    message = st.chat_input("Ask a question to the website.")
    if message:
        send_message(message, "human")
        chain = (
            {
                "docs": retriever, 
                "question": RunnablePassthrough(),
            } 
            | RunnablePassthrough.assign(api_key=lambda _: api_key)
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )

        with st.chat_message("ai"):
            response = chain.invoke(message)

