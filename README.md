# 📚 Masal Diyarı Bilgi Asistanı (RAG Chatbot)

Bu proje, temel masal metinlerini (PDF ve TXT formatında) kullanarak doğal dil sorularına cevap verebilen bir RAG (Retrieval-Augmented Generation - Geri Getirme Artırımlı Üretim) sistemidir. Kullanıcılar, yüklenen masalların içeriği hakkında sorular sorabilir ve yapay zeka bu metinlere dayanarak cevaplar üretir.

## ⚙️ Kullanılan Teknolojiler

Bu proje, güncel Yapay Zeka (AI) ve Veri Bilimi araçlarının en iyi kombinasyonunu kullanır:

| Teknoloji | Amaç |
| :--- | :--- |
| **Python** | Projenin temel programlama dili. |
| **Streamlit** | Projeyi web arayüzüne (Chatbot) dönüştürme. |
| **LangChain** | LLM'ler ve veritabanı (RAG zinciri) arasındaki etkileşimi yönetme. |
| **Google Gemini API** | Gelişmiş dil anlama ve cevap üretme (Gemini-2.5-Flash). |
| **Sentence-Transformers** | Verileri lokal olarak vektörlere çevirme (Kotadan bağımsız çalışma). |
| **`langchain-community`** | PDF ve TXT dosyalarını yükleme ve işleme. |

## 🚀 Kurulum ve Çalıştırma

Projenin yerel bilgisayarınızda çalışması için aşağıdaki adımları takip edin.

### 1. Ortam Kurulumu

Öncelikle bir Conda ortamı oluşturun ve aktifleştirin:

```bash
conda create -n masal-conda python=3.10 -y
conda activate masal-conda
