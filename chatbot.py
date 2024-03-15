from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st

load_dotenv()
# Khởi tạo url để lấy thông tin
urls = [
    "https://cayxanhsaigonvn.com/tin-tuc/cach-chon-cay-trong-van-phong-cho-khong-gian-lam-viec-them-xanh-mat-22.html",
    "https://locat.com.vn/blog/434562_cach-chon-loai-cay-xanh-phu-hop-tao-khong-gian-lam-viec-xanh"]
# thu thập thông tin từ website
loader = WebBaseLoader(urls)
data = loader.load()
print(data)

# chia dữ liệu thành các đoạn để embeddings
text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=10000,
                                      chunk_overlap=200)
docs = text_splitter.split_documents(data)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in docs)
texts = text_splitter.split_text(context)


# Xử lí dữ liệu, bỏ các dấu xuống dòng
def process_chunks(chunks):
    processed_chunks = []  # Danh sách để lưu trữ các chunk đã được xử lý
    for chunk in chunks:  # Duyệt qua từng chunk trong danh sách chunks
        lines_combined = ""  # Chuỗi để lưu trữ các dòng được nối thành một đoạn string
        for line in chunk.split('\n'):  # Duyệt qua từng dòng trong chunk
            line = line.strip()  # Loại bỏ khoảng trắng ở đầu và cuối dòng
            if line:  # Kiểm tra xem dòng có dữ liệu hay không
                lines_combined += line + " "  # Nối dòng vào chuỗi kết quả, kết thúc bằng khoảng trắng
        processed_chunks.append(
            lines_combined.strip())  # Thêm chuỗi đã xử lý vào danh sách các chunk đã xử lý, loại bỏ khoảng trắng ở cuối
    return processed_chunks


for p in process_chunks(texts):
    print(p)
# Tạo model
google_api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key,
                               temperature=0.4, convert_system_message_to_human=True)
# embeddings dữ liệu
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
# Lưu vector vào chromadb
vector_index = Chroma.from_texts(process_chunks(texts), embeddings).as_retriever(search_kwargs={"k": 2})

# prompts truy vấn
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""

# Mô hình truy vấn
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  # Run chain
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


# Giao diện
def generate_response(question):
    result = qa_chain({"query": question})
    return result['result']


st.title("Chatbot")
if "message" not in st.session_state:
    st.session_state.message = []
for message in st.session_state.message:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Nhập câu hỏi"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.message.append({"role": "user", "content": question})
    with st.chat_message("assistant"):
        st.markdown(generate_response(question))
    st.session_state.message.append({"role": "assistant", "content": generate_response(question)})
