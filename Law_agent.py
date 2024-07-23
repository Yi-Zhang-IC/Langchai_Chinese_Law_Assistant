from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from langserve import add_routes
from dotenv import load_dotenv
import uvicorn
import bs4
import os

# load civil code of China
def load_documents():
    loader = WebBaseLoader(
        web_path="https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm",
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
    )
    docs = loader.load()
    print("Documents loaded successfully.")
    return docs

# split the documents into smaller chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print("Documents split into chunks successfully.")
    return splits

# store the documents in a vector store
def store_documents(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    print("Documents stored in vector store successfully.")
    return vectorstore

# build a RAG chain
def build_rag_chain(vectorstore):
    prompt_template_str = """
# 你是一个精通中国民法典的法律顾问。你的客户是一位想要了解中国民法典的普通公民。
# 当您被问到一个法律问题时，您会进行以下分析：
# 1. 根据事实细节分析问题。
# 2. 指出法律问题的重点。
# 3. 根据中国民法典的相关法条以及法条所处的章节为事件定性，并且给出解释。
# 4. 为客户提供建议。

# 请回答以下问题：{question}
# 背景知识：{context}
# 请你严格遵照背景知识回答问题。不要更改原始的信息。如果背景知识中没有提到的信息，请回答“我不知道”。
# 在你回答的时候，遵循以下的原则：
# 1. 保持客观中立。
# 2. 不要提供任何个人信息。
# 3. 不要提供任何法律建议。
# 4. 不要提供任何不符合中国民法典的信息。
# 并且按照以下格式输出：

问题: {question}

分析:
1. 问题的事实细节分析
2. 法律问题的重点
3。事件的定性及解释

定性及解释:
- 定性这个事件是什么类型的事件，并依据法条详细解释。依据情节是否严重分情况讨论。
- 定性这个事件所参考的法条，以及所属章节: 请完整的写出法条。
- 这个行为导致了什么结果，如何衡量这个结果，分情况讨论。
- 衡量这个结果所参考的法条，所属章节， 请完整的写出法条。
- 进一步解释这个结果，例如如果涉及到赔偿，如何计算赔偿金额，分情况讨论。
- 计算赔偿金额所参考的法条，所属章节， 请完整的写出法条。

建议:
- 给出你的结论
- 提供中立的建议
    """
    prompt_template = PromptTemplate.from_template(prompt_template_str)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    # Load environment variables from .env file
    load_dotenv()

    # Initialize the OpenAI model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.8,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    print("RAG chain built successfully.")
    return rag_chain

def create_app(rag_chain):
    # Create a FastAPI app
    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="AI Agent in Law",
    )

    # Add routes to the FastAPI app
    add_routes(
        app,
        rag_chain,
        path="/first_llm",
    )
    return app

def main():
    # Load the documents
    docs = load_documents()

    # Split the documents
    splits = split_documents(docs)

    # Store the documents in a vector store
    vectorstore = store_documents(splits)

    # Build a RAG chain
    rag_chain = build_rag_chain(vectorstore)

    # Create a FastAPI app
    app = create_app(rag_chain)

    if __name__ == "__main__":
        uvicorn.run(app, host="localhost", port=8000)   

main()