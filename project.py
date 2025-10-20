# -----------------------------------------------------------
# 1. Kütüphane İçe Aktarma ve Ayarlar (Lokal Vektörleme Eklendi)
# -----------------------------------------------------------
import os
import glob
from dotenv import load_dotenv 
import streamlit as st

# Lokal Embedding Modeli için kütüphane (API limitini aşar)
from langchain_community.embeddings import SentenceTransformerEmbeddings 

# Diğer LangChain Bileşenleri
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader 
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA

# .env dosyasındaki GOOGLE_API_KEY'i yükle
load_dotenv() 

st.set_page_config(layout="wide")

# -----------------------------------------------------------
# 2. RAG Pipeline Kurulumu (Lokal Vektörleme)
# -----------------------------------------------------------

@st.cache_resource
def setup_rag_pipeline():
    st.info("Veriler Yükleniyor ve LOKAL OLARAK Vektörleniyor. Lütfen Bekleyin...")

    # Dosya Yükleme (PDF ve TXT)
    txt_files = glob.glob("masallar/*.txt")
    pdf_files = glob.glob("masallar/*.pdf")
    documents = []
    
    # Yükleme işlemleri...
    for path in txt_files:
        try:
            loader = TextLoader(path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"TXT yüklenirken hata: {path}, Hata: {e}")

    for path in pdf_files:
        try:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"PDF yüklenirken hata: {path}, Hata: {e}")

    # 1. Metin Parçalama (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # 2. LOKAL Vektörleme (Embedding) - ARTIK API KOTASINI AŞMAYACAK!
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Vektör Veritabanı
    vectorstore = DocArrayInMemorySearch.from_documents(texts, embedding_model)
    
    st.success(f"Veri Yükleme ve Lokal Vektörleme Tamamlandı. Parça Sayısı: {len(texts)}")
    
    # 4. LLM ve RetrievalQA Zinciri (Sadece cevap üretimi için Gemini API'si kullanılır)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
    )
    
    return qa_chain

# RAG Pipeline'ı kur
qa_chain = setup_rag_pipeline()


# -----------------------------------------------------------
# 3. Streamlit Arayüzü ve Çalışma Mantığı
# -----------------------------------------------------------

st.title("📚 Masal Diyarı Bilgi Asistanı (Lokal Vektörleme)")
st.markdown("Merhaba! Lokal vektörleme kullanarak çalışan gelişmiş RAG asistanıdır.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hangi masal hakkında soru sormak istersiniz?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Masal Diyarı'nda cevap aranıyor..."):
        try:
            # Sadece tek bir çağrı yapıyoruz: QA zincirini çağırma
            result = qa_chain.invoke({"query": prompt})
            generated_text = result['result']
            source_info = "\n\n**Cevap, yüklenen masal metinlerine dayanılarak üretilmiştir.**"
            full_response = generated_text + source_info

        except Exception as e:
            full_response = f"Üzgünüm, bir hata oluştu. Lütfen API Anahtarınızı ve internet bağlantınızı kontrol edin. Hata: {e}"
            
    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
