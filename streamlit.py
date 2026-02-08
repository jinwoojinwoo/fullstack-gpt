from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler 
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda

import streamlit as st
import json
from pathlib import Path



st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

Path("./.cache/quiz_files").mkdir(parents=True, exist_ok=True)

st.title("QuizGPT")

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)





@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,   
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)    
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, api_key, difficulty):
    function = {
        "name": "create_quiz",
        "description": "function that takes a list of questions and answers and returns a quiz",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                            },
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {
                                            "type": "string",
                                        },
                                        "correct": {
                                            "type": "boolean",
                                        },
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                        "required": ["question", "answers"],
                    },
                }
            },
            "required": ["questions"],
        },
    }
    
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
        openai_api_key=api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ]
    )

    questions_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
                
            Based ONLY on the following context make 2 questions to test the user's knowledge about the text.

            You will receive a Difficulty Level indicating the quiz difficulty. 
            The value will be either ‘easy’ or ‘hard’.

            Based on the difficulty level, generate quiz questions following these rules:

            If difficulty is "easy":
            - Create simple, straightforward questions that test basic understanding.
            - Avoid tricky or ambiguous wording.
            - Make the correct answer clearly distinguishable.

            If difficulty is "hard":
            - Create challenging questions that require reasoning or careful reading.
            - Use nuanced and closely related answer choices.
            - Avoid any wording that reveals the correct answer.

            Each question should have 4 answers, three of them must be incorrect and one should be correct.
                
            Use (o) to signal the correct answer.
                
            Question examples:
                
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
                
            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
                
            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998
                
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model
                
            Your turn!

            Difficulty Level: {difficulty}
                
            Context: {context}
            """,
        )
    ])


    questions_chain = {"difficulty": RunnableLambda(lambda _: difficulty), "context": format_docs} | questions_prompt | llm 

    response = questions_chain.invoke(_docs)
    return json.loads(response.additional_kwargs["function_call"]["arguments"])


@st.cache_data(show_spinner="Searching wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs
            

with st.sidebar:
    topic = None
    docs = None

    api_key = st.text_input(
        "OpenAI API Key",
    )

    difficulty = st.selectbox(
        "Choose a quiz level",
        (
            "easy",
            "hard",
        ),
    )
        
    choice = st.selectbox(
        "Choose what you want to use", 
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["docx", "txt", "pdf"],
        )

        if file:
            docs = split_file(file)
    elif choice == "Wikipedia Article":
        topic = st.text_input("Name of the article")
        if topic:
            docs = wiki_search(topic)

    st.link_button("github", "https://github.com/jinwoojinwoo/fullstack-gpt")
         
if not docs:
    st.markdown(
        """
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name, api_key, difficulty)
    with st.form("questions_form"):
        all_correct = True
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if not value:
                all_correct = False
                continue
            elif {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            else:
                all_correct = False
                st.error("Wrong!")

        if all_correct:
            st.balloons()

        button = st.form_submit_button(disabled=True if all_correct else False)

        

        
