# 📚 Masal Diyarı Bilgi Asistanı (RAG Chatbot)

## Proje Tanımı ve Amacı

Bu proje, belirlenen masal metinlerini (PDF ve TXT formatında) kullanarak doğal dil sorularına kanıta dayalı cevaplar üreten, **Retrieval-Augmented Generation (RAG)** mimarisine dayalı bir Chatbot uygulamasıdır. 

Uygulama, lokal vektörleme (embedding) kullanarak Google API kotasına takılmadan çalışırken, cevap üretimi için Gemini'nin güçlü Büyük Dil Modeli (LLM) yeteneklerini kullanır. Projenin amacı, yapay zekanın sadece genel bilgiye değil, aynı zamanda özel ve kısıtlı veri setlerine (Masal Diyarı) dayalı olarak da doğru ve alakalı cevaplar üretebildiğini göstermektir.

---

## ⚙️ Kullanılan Teknolojiler

Bu RAG projesi, en güncel ve sektör standardı araçları kullanarak geliştirilmiştir.

| Teknoloji | Türü | Amaç ve Rolü |
| :--- | :--- | :--- |
| **Python 3.10+** | Programlama Dili | Projenin temel geliştirme ortamı. |
| **Streamlit** | Web Arayüzü | Chatbot arayüzünün oluşturulması ve kullanıcı etkileşiminin sağlanması. |
| **LangChain** | Çatı (Framework) | RAG zincirinin (LLM, Embeddings, Vektör Veritabanı) yönetimi ve koordinasyonu. |
| **Google Gemini API** | LLM | Kullanıcının sorusuna nihai, akıcı ve anlamlı cevabı üretme (Model: `gemini-2.5-flash`). |
| **Sentence-Transformers** | Embedding Model | Metinleri yüksek boyutlu vektörlere çevirme (Model: `all-MiniLM-L6-v2`). **Lokal** çalışarak API maliyetini ve kotasını ortadan kaldırır. |
| **Pypdf & LangChain Loaders** | Veri İşleme | PDF ve TXT gibi yapılandırılmamış masal verilerinin uygulamaya yüklenmesi. |
| **Conda / Pip** | Paket Yöneticisi | Proje bağımlılıklarının yönetilmesi ve ortamın izole edilmesi. |

---

## 🧠 Çözüm Mimarisi (RAG Zinciri)

Proje, iki aşamalı güçlü bir Bilgi Alma Artırımlı Üretim (RAG) mimarisini takip eder:

1.  **Veri Yükleme ve Parçalama (Ingestion):**
    * **Girdi:** `masallar/` klasöründeki tüm PDF ve TXT dosyaları.
    * **Parçalama:** `RecursiveCharacterTextSplitter` kullanılarak metinler anlam bütünlüğünü koruyacak şekilde küçük parçalara (chunks) ayrılır.
2.  **Lokal Vektörleme ve Veritabanı (Indexing):**
    * **Embedding:** Parçalanan her metin parçası, **lokal olarak** çalışan `all-MiniLM-L6-v2` modeli ile sayısal vektörlere çevrilir.
    * **Veritabanı:** Bu vektörler, bellekte (in-memory) tutulan `FAISS` Vektör Veritabanına kaydedilir.
3.  **Sorgu ve Cevap Üretimi (Generation):**
    * **Sorgu Vektörleme:** Kullanıcının sorduğu soru, aynı lokal modelle vektöre çevrilir.
    * **Alaka Düzeyi Arama (Retrieval):** Vektör veritabanında, kullanıcının sorusuna en çok benzeyen (en alakalı) metin parçaları hızlıca bulunur.
    * **Üretim (Generation):** Bulunan bu alakalı metin parçaları, Gemini-2.5-Flash modeline gönderilir. Model, bu parçaları referans alarak sorunun cevabını üretir.

---

## 🚀 Kurulum ve Proje Yapısı

### Proje Yapısı
