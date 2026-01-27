import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Page configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Simple Chatbot")

# Define available models and their providers
MODELS = {
    "gpt-4.1-nano": "openai",
    "gpt-5.1-nano": "openai",
    "gemini-2.5-flash-lite": "google_genai",
    # "gemini-pro": "google_genai",
    "smollm2:1.7b": "ollama",
}


# Rendering
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Model Name", options=list(MODELS.keys()))
    provider = MODELS[selected_model]
    st.caption(f"Provider: {provider}")

    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
    max_tokens = st.number_input("Max Tokens", min_value=5, max_value=4096, value=512)

    st.divider()
    memory_window = st.number_input("Memory Window (messages)", min_value=1, max_value=1000, value=20)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

llm = init_chat_model(selected_model, model_provider=provider, temperature=temperature, max_tokens=max_tokens, streaming=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages_to_send = st.session_state.messages[-memory_window:]
        stream = llm.stream(messages_to_send)
        response = st.write_stream(stream)

    st.session_state.messages.append(AIMessage(content=response))
