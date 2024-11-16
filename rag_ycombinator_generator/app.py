import os

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv('../.env')

# Set environment variables (replace with your actual keys or set them in your environment)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Load data from CSV
df = pd.read_csv("database/2023-07-13-yc-companies.csv", usecols=['long_description'])
df = df.drop_duplicates()

# Fill NaNs and convert to strings
df['long_description'] = df['long_description'].fillna("").astype(str)

# Convert data into documents
docs = [Document(page_content=row['long_description']) for _, row in df.iterrows()]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create vectorstore
persist_directory = 'chroma_db'

if not os.path.exists(persist_directory):
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
else:
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings()
    )

# Get retriever
retriever = vectorstore.as_retriever()

# Create prompt template
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={'prompt': prompt}
)

# Set up FastAPI app
app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    response = qa_chain.invoke(request.question)
    return {"answer": response['result']}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
