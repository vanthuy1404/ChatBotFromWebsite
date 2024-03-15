from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

from langchain_text_splitters import CharacterTextSplitter

google_api_key = os.getenv("GOOGLE_API_KEY")
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=google_api_key,
                             temperature=0.4,convert_system_message_to_human=True)
urls =["https://cayxanhsaigonvn.com/tin-tuc/cach-chon-cay-trong-van-phong-cho-khong-gian-lam-viec-them-xanh-mat-22.html"]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
print(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
context = "\n\n".join(str(p.page_content) for p in data)
texts = text_splitter.split_text(context)
print(texts)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api_key)
vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":2})

# qa_chain = RetrievalQA.from_chain_type(
#     model,
#     retriever=vector_index,
#     return_source_documents=True
#
# )
question = input("Nhap vao cau hoi cua ban: ")
# result = qa_chain({"query": question})
# # result["result"]
# print(result['result'])
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
result = qa_chain({"query": question})
print(result['result'])