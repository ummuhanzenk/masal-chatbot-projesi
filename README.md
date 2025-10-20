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

## 🚀 Kurulum ve Çalıştırma Kılavuzu

Bu proje, Python'ın sanal ortamı olan Conda kullanılarak izole bir ortamda çalıştırılmak üzere tasarlanmıştır.

### 1. Ön Koşullar

1.  **Python:** Python 3.10 veya üzeri kurulu olmalıdır.
2.  **Conda/Miniconda:** Sanal ortam yönetimi için Conda veya Miniconda kurulu olmalıdır.
3.  **Gemini API Anahtarı:** Google AI Studio'dan edinilmiş geçerli bir API anahtarı.

### 2. Proje Dosyaları ve Yapısı

Projenin anlaşılırlığını artırmak için aşağıdaki temel dizin yapısı kullanılır:
### 3. Ortam Kurulum Adımları

Projenizi yerel bilgisayarınızda (Windows, Mac, Linux) çalıştırmak için aşağıdaki adımları sırasıyla uygulayın.

1.  **Depoyu Klonlama:**
    ```bash
    git clone [https://github.com/ummuhanzenk/masal-chatbot-projesi.git](https://github.com/ummuhanzenk/masal-chatbot-projesi.git)
    cd masal-chatbot-projesi
    ```

2.  **Conda Ortamı Oluşturma ve Aktifleştirme:**
    ```bash
    conda create -n masal-conda python=3.10 -y
    conda activate masal-conda
    ```

3.  **Bağımlılıkları Yükleme:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Anahtarını Yapılandırma:**
    * Proje klasörünün içinde **`.env`** adında bir dosya oluşturun.
    * Dosyanın içine kendi Gemini API anahtarınızı ekleyin (Örnek satır):
        ```env
        # .env dosyası içeriği:
        GOOGLE_API_KEY="SİZİN_API_ANAHTARINIZ"
        ```

5.  **Uygulamayı Başlatma:**
    ```bash
    python -m streamlit run project.py
    ```
    *Uygulama, yerel tarayıcınızda otomatik olarak açılacaktır. İlk çalıştırmada lokal embedding modelini indirir.*

---

## 🌐 Web Arayüzü ve Ürün Kılavuzu

Proje, kullanıcı dostu bir Streamlit arayüzü ile sunulmaktadır.

### Ürün Özellikleri:

* **Canlı Sohbet Arayüzü:** Kullanıcıların sorularını girmesi ve cevapları görmesi için modern bir Chatbot penceresi sunar.
* **Lokal Vektörleme Bildirimi:** Uygulama başlatılırken, verilerin lokal olarak işlendiği bilgisi ekranda gösterilerek şeffaflık sağlanır.
* **Otomatik Yükleme:** `masallar/` klasörüne yeni bir masal dosyası eklendiğinde, uygulama yeniden başlatıldığında otomatik olarak öğrenme sürecine dahil edilir.
* **Veri Seti Bilgisi:** Arayüz, kaç adet metin parçasının (chunks) işlendiğini göstererek kullanıcının bilgi kaynağının büyüklüğünü anlamasını sağlar.

### Kullanım Kılavuzu:

1.  Uygulama başlatıldığında, öncelikle tüm masal metinleri yüklenir ve vektörlenir (İlk çalıştırmada bu biraz zaman alabilir).
2.  İşlem tamamlandığında sohbet kutusu aktif hale gelir.
3.  **Soru Sorun:** "Kırmızı Başlıklı Kız'ın sepetinde ne vardı?" gibi, yüklediğiniz masallarla ilgili bir soru yazın ve Enter tuşuna basın.
4.  LLM (Gemini-Flash), ilgili masal parçalarını kullanarak size kanıta dayalı bir cevap sunacaktır.
