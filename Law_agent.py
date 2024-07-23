import gradio as gr
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

# Load civil code of China
def load_documents():
    loader = WebBaseLoader(
        web_path="https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm",
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
    )
    docs = loader.load()
    print("Documents loaded successfully.")
    return docs

# Split the documents into smaller chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print("Documents split into chunks successfully.")
    return splits

# Store the documents in a vector store
def store_documents(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    print("Documents stored in vector store successfully.")
    return vectorstore

# Build a RAG chain
def build_rag_chain(vectorstore):
    prompt_template_str = """
    你是一个精通中国民法典的法律顾问。你的客户是一位想要了解中国民法典的普通公民。当你被问到一个法律问题时，你首先会进行以下分析：
        1. 问题的事实细节分析，简要描述问题中的关键事实。
        2. 法律问题的重点， 阐明与问题相关的主要法律问题。
        3. 定性这个事件是什么类型的事件，并依据法条详细解释。依据情节是否严重分情况讨论。定性这个事件所参考的法条，以及所属章节，请完整的写出法条。
        4. 这个行为导致了什么结果，如何衡量这个结果，分情况讨论。衡量这个结果所参考的法条，所属章节，请完整的写出法条。
        5. 进一步解释这个结果，例如如果涉及到赔偿，如何计算赔偿金额，分情况讨论。计算赔偿金额所参考的法条，所属章节，请完整的写出法条。
        6. 结果分析，分析该行为可能导致的结果及其衡量标准。引用相关法条和章节，讨论不同情境下的结果。
        7. 建议，给出你的结论并且提供几条中立的建议。
    
    分析完成之后，你会根据你的分析内容回答问题。在你的回答中，请不要重复上面的思路。你需要给出一个结构清晰，易于阅读，语言流畅的回答。在你的回答中，请使用小标题的形式（但不要用markdown）让读者理解你每一部分的内容以及思路。在你引用法条的时候，请空出一行，并使用双引号强调。
    范例回答:
    - 初步分析问题
        - xxx
        - 相关法条1："xxx"
        - 相关法条2："xxx"
        - ........(如果有法条3)
    - 问题重点以及细节
        - xxx
        - 相关法条1："xxx"
        - 相关法条2："xxx"
        - ........(如果有法条3)
    - 事件定性以及依据
        - xxx
        - 相关法条1："xxx"
        - 相关法条2："xxx"
        - ........(如果有法条3)
    - 结果衡量以及依据
        - xxx
        - 相关法条1："xxx"
        - 相关法条2："xxx"
        - ........(如果有法条3)
    - 分情况讨论结果
        - xxx
        - 相关法条1："xxx"
        - 相关法条2："xxx"
        - ........(如果有法条3)
    - 建议：
        1. xxx
        2. xxx
        3. xxx

    请回答以下问题：{question}
    背景知识：{context}
    请你严格遵照背景知识回答问题。在你回答的时候请不要重复上面的思路。如果背景知识中没有提到的信息，请回答“我不知道”。 在你回答的时候，遵循以下的原则：

    1. 保持客观中立。
    2. 不要提供任何个人信息。
    3. 不要提供任何法律建议。
    4. 不要提供任何不符合中国民法典的信息。
    5. 你的回答不能按照markdown解析。请考虑使用标点符号和换行来使你的回答更易于阅读。

    """
    prompt_template = PromptTemplate.from_template(prompt_template_str)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    # Load environment variables from .env file
    load_dotenv()

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=1,
        max_tokens=2000,
        top_p=1,
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

def handle_query(question):
    # Assuming `rag_chain` is accessible here, either through import or as a global variable
    response = rag_chain.invoke(question)
    return response

def main():
    # Load the documents
    docs = load_documents()

    # Split the documents
    splits = split_documents(docs)

    # Store the documents in a vector store
    vectorstore = store_documents(splits)

    # Build a RAG chain
    global rag_chain
    rag_chain = build_rag_chain(vectorstore)

    # Create a Gradio interface
    iface = gr.Interface(
        fn=handle_query, 
        inputs="text", 
        outputs="text",
        title="智能AI法律顾问——民法",
        description="您好！我是智能AI法律顾问，我精通中国民法典。请在下方输入您的问题，我将为您提供专业的法律咨询。"
    )

    # Launch Gradio interface with sharing enabled
    iface.launch(share=True)

if __name__ == "__main__":
    main()
