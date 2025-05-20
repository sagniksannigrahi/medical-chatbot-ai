from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_core.runnables import RunnableMap
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

retriever = None
rag_chain = None

def get_rag_chain():
    global retriever, rag_chain
    if rag_chain is None:
        embeddings = download_hugging_face_embeddings()
        docsearch = PineconeVectorStore.from_existing_index(
            index_name="medicalbot",
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.4, max_tokens=500)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = prompt | llm
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


index_name = "medicalbot"


#question_answer_chain = create_retrieval_chain(llm, prompt)
#rag_chain = create_retrieval_chain(retriever, question_answer_chain)






@app.route("/")
def home():
    return "Medical Chatbot is live!"
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("Received message:", msg)

    try:
        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"].content if hasattr(response["answer"], "content") else str(response["answer"])

    except Exception as e:
        print("Error occurred:", str(e))
        answer = "I'm currently unable to answer your question due to a technical issue. Please try again later."

    return str(answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
