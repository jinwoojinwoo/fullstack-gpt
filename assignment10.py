from bs4 import BeautifulSoup
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
import os
import requests
from openai import OpenAI
import streamlit as st
import time
import json

def search_in_wikipedia(inputs):
    query = inputs["query"]
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)


def search_in_duckduckgo(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(query)


def scrap_web_page(inputs):
    url = inputs["url"]

    try: 
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ScrapWebPageFunc/1.0)"}
        r = requests.get(url, headers=headers)
        r.raise_for_status()

        soup = BeautifulSoup(r.text)

        for tag in soup(["header", "footer", "nav"]):
            tag.decompose()

        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text(separator="\n", strip=True)

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)
    except requests.exceptions.HTTPError as e:
        return ""
    
def save_file(inputs):
    filename = inputs["filename"]
    content = inputs["content"]

    filepath = f"./{filename}"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")
    return f"{filepath} created"

functions_map = {
    "search_in_wikipedia": search_in_wikipedia,
    "search_in_duckduckgo": search_in_duckduckgo,
    "scrap_web_page": scrap_web_page,
    "save_file": save_file,
}


functions = [
    {
        "type": "function",
        "function": {
            "name": "search_in_wikipedia",
            "description": "Given a research query, returns the corresponding Wikipedia search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Query you will search for. Example: Apple's products",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_duckduckgo",
            "description": "Given a research query, returns the corresponding Duckduckgo search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Query you will search for. Example: Apple's stock",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrap_web_page",
            "description": "Given a URL to collect, returns the text content with HTML tags removed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the webpage to scrape. Example: https://www.apple.com",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "Given a filename and content, saves the content to the specified file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename to create. Example: sample.txt",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to save to the file.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]


# assistant = client.beta.assistants.create(
#     name="Resaercher Assistant",
#     instructions="You help users research a given topic, and you help users save the research results to a file.",
#     model="gpt-4.1-mini",
#     tools=functions,
# )

assistant_id="asst_K8r8uITQkgcRVHOnCID1zHdm"


def get_run(client, run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def send_message(role, message, client, thread_id, assistant_id, run=False):
    with st.chat_message(role):
        st.markdown(message)

    if run:
        client.beta.threads.messages.create(
            thread_id=thread_id, role=role, content=message
        )

        return client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
            )
    else:
        return None
    
def get_messages(client, thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages

def print_history(client, thread_id, from_n=0):
    messages = get_messages(client, thread_id)
    for message in messages[from_n:]:
        send_message(
            message.role,
            message.content[0].text.value,
            client, 
            thread_id, 
            None,
            run=False,
        )
    return len(messages)

def get_tool_outputs(client, run_id, thread_id):
    run = get_run(client, run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(client, run_id, thread_id):
    outputs = get_tool_outputs(client, run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )

with st.sidebar:
    api_key = st.text_input(
        "OpenAI API Key",
    )

    st.link_button("github", "https://github.com/jinwoojinwoo/fullstack-gpt")


if api_key:
    client = OpenAI(api_key=api_key)

    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state["thread_id"] = thread.id

    msg_num = print_history(client, st.session_state["thread_id"])

    message = st.chat_input("Ask something..")
    if message:
        run = send_message("user", message, client, st.session_state["thread_id"], assistant_id, run=True)
        while get_run(client, run.id, st.session_state["thread_id"]).status in ('queued', 'in_progress'):
            st.write(get_run(client, run.id, st.session_state["thread_id"]).status)
            time.sleep(1)
        st.write(get_run(client, run.id, st.session_state["thread_id"]).status)
        # st.write(get_messages(client, st.session_state["thread_id"]))

        if get_run(client, run.id, st.session_state["thread_id"]).status == 'requires_action':
            submit_tool_outputs(client, run.id, st.session_state["thread_id"])

            while get_run(client, run.id, st.session_state["thread_id"]).status in ('queued', 'in_progress'):
                st.write(get_run(client, run.id, st.session_state["thread_id"]).status)
                time.sleep(1)

        print_history(client, st.session_state["thread_id"], msg_num + 1)
        


        


