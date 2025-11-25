from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# packages:
# pip install -U langchain
# pip install -U langchain-openai
# pip install -U langchain-google-genai
# pip install dotenv
# pip install streamlit

load_dotenv()

model = init_chat_model("gpt-4.1-nano", model_provider="openai", temperature=0, max_tokens=10)
# model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", temperature=0, max_tokens=20)
# model = init_chat_model("smollm2:1.7b", model_provider="ollama", temperature=0, max_tokens=5)

response = model.invoke("Why do parrots have colorful feathers?")
print(response)
# print(type(response))
#
#
conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to Catalan."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]
#
# # using abstractions
# conversation = [
#     SystemMessage("You are a helpful assistant that translates English to Catalan."),
#     HumanMessage("Translate: I love programming."),
#     AIMessage("J'adore la programmation."),
#     HumanMessage("Translate: I love building applications.")
# ]
#
response = model.invoke(conversation)
print(response)
print(type(response))
#
# minor




