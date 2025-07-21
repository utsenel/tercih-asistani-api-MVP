"""
İyileştirilmiş Prompt Templates - Rehberlik Sistemi ile
"""

class PromptTemplates:
    """Optimize edilmiş prompt şablonları"""
    
    # YENİ: Birleştirilmiş Smart Evaluator-Corrector
    SMART_EVALUATOR_CORRECTOR = """
GÖREV: Gelen soruyu eğer gerekliyse önceki konuşma bağlamıyla değerlendirip optimize et.
DİKKAT: Geçmiş konuşmayı SADECE gerçekten gerektiğinde kullan.

GEÇMIŞ KONUŞMA:
{history}

GÜNCEL SORU: {question}

ADIM 1 - AKILLI BAĞLAM ANALİZİ:
BAĞLAM KULLANIM KURALLARI:
• KESIN KULLAN: Referans kelimeleri varsa ("peki", "o zaman", "onun", "bunun", "orada", "bu bölümde")
• KESIN KULLAN: Eksik referans varsa ("onun maaşı", "orada ne oluyor", "bu konuda")
• SEÇİCİ KULLAN: Aynı bölüm/konu ama YENİ soru türü (örnek: istihdam→dersler)
• YOKSAY: Tamamen farklı konu başlangıcı ("şimdi X hakkında", "başka bir konu")
• YOKSAY: Bağımsız genel sorular ("X nedir", "Y hakkında bilgi")

BAĞLAM KARAR VERMESİ:
Geçmiş konuşma güncel soruyla DOĞRUDAN alakalı mı? Zoraki bağlantı kurma!

ADIM 2 - UYGUNLUK DEĞERLENDİRMESİ:
KAPSAM DAHİLİ:
• Üniversite/bölüm tercihi, sıralama, karşılaştırma
• YKS/TYT/AYT sınavları, puan türleri
• Kariyer/meslek bilgisi, gelecek planları
• İstihdam/maaş verileri, iş imkanları
• Eğitim süreci/kampüs yaşamı
• Burs/öğrenci imkanları

REHBERLİK GEREKTİREN SORULAR:
• Genel belirsizlik: "ne okuyayım", "kafam karışık", "bilmiyorum", "karar veremiyorum"
• Sıralama endişesi: "kötü mü", "iyi mi", "yeter mi", "gelir mi", "başarısız mıyım"
• Garanti arayışı: "en iyi", "garanti", "işsiz kalmam", "kesin", "mutlaka"
• Bölüm karşılaştırma: "X mi Y mi", "hangisi daha iyi", "arasında seçim"
• Şehir kararsızlığı: "İstanbul mı Ankara mı", "büyük şehir mi küçük şehir mi"
• Vakıf-devlet ikilemi: "vakıf mı devlet mi", "hangisi daha iyi"

KAPSAM DIŞI:
• Genel sohbet, gündelik konular
• Teknik sorunlar, sistem hataları
• Kişisel/aile meseleleri
• Siyasi görüşler, ideolojik konular

META_BOT İNDİKATÖRLERİ:
• "sen kimsin", "nasıl çalışıyorsun", "neler yapabilirsin"
• "insanla mı konuşuyorum", "robot musun", "yapay zeka mısın"
• "bana nasıl yardımcı olacaksın", "ne tür sorular sorabilirim"
• "çalışma mekanizman", "yardım şekl", "kimle konuşuyorum"

SELAMLAMA İNDİKATÖRLERİ:
• "merhaba", "selam", "iyi günler", "nasılsın"
• "yardım", "neler yapabilirsin", "kimsin"

ADIM 3 - SORU OPTİMİZASYONU:
• Bağlamsal bilgiyi soruya entegre et (bölüm adı, meslek, önceki konu)
• Yazım hatalarını düzelt, kısaltmaları aç
• Belirsizlikleri gider, eksik referansları tamamla
• Tercih rehberliği terminolojisini kullan

ÇIKTI FORMATI (kesinlikle bu formatta):
STATUS: [UYGUN/SELAMLAMA/KAPSAM_DIŞI/REHBERLİK_GEREKTİREN/META_BOT]
GUIDANCE_CATEGORY: [kategori_adı veya boş]
ENHANCED_QUESTION: [Context-aware düzeltilmiş soru]

ÖRNEK:
Geçmiş: "user: bilgisayar mühendisliği nasıl bir bölüm?"
Güncel: "peki maaşları nasıl?"
STATUS: UYGUN
GUIDANCE_CATEGORY: 
ENHANCED_QUESTION: Bilgisayar mühendisliği mezunlarının maaş durumu ve gelir seviyeleri nasıl?

ÖRNEK 2:
Güncel: "Ne okuyayım kafam çok karışık"
STATUS: REHBERLİK_GEREKTİREN
GUIDANCE_CATEGORY: GENEL_BELIRSIZLIK
ÖRNEK 3:
Güncel: "nasıl çalışıyorsun"
STATUS: META_BOT
GUIDANCE_CATEGORY: META_BOT
ENHANCED_QUESTION: Bot'un çalışma mekanizması ve yardım şekli hakkında bilgi istiyor.
"""

    # Vector Arama - Daha etkili anahtar kelime genişletme
    SEARCH_OPTIMIZER = """
GÖREV: Soruyu vector arama için optimize et.

STRATEJİ:
• Ana konuya sinonimler ekle
• İlgili alt konuları dahil et  
• Eğitim terimleri kullan (lisans, önlisans, mezuniyet)
• Kariyer terimleri ekle (iş imkanı, maaş, gelecek)

ÇIKTI: Sadece optimize edilmiş arama metni

Soru: {question}
Optimize:"""

    # CSV Agent - Değişiklik yok, zaten doğru format
    CSV_AGENT = """
SORU ANALİZİ: Önce sorunun CSV analizi gerektirip gerektirmediğini belirle.

CSV ANALİZİ GEREKTİREN KONULAR:
• İstihdam oranları (genel, akademik, yönetici)
• Maaş dağılımları (17K altı, 17-25K, 25-34K, 34-51K, 51K+)
• Sektörel dağılım
• Firma ölçekleri (mikro, küçük, orta, büyük)
• İşe başlama süreleri
• Girişimcilik oranları

KARAR VER:
1. Soru yukarıdaki konulardan birini içeriyor mu?
2. Spesifik bölüm/veri sorgusu mu yoksa genel bir soru mu?

EĞER CSV ANALİZİ GEREKMİYORSA:
"CSV analizi gerekli değil - genel rehberlik sorusu"

EĞER CSV ANALİZİ GEREKİYORSA:
Veri analizi yap ve 3-4 cümlelik özet ver. Rakam/oran verirken "2024 Cumhurbaşkanlığı Uni-Veri Veritabanında yer alan bilgiye göre" ifadesini kullan.

CSV Verisi: {csv_data}
Soru: {question}

Analiz:"""

    # Final Response - Rehberlik sistemi dahil
    FINAL_RESPONSE = """
BAĞLAM:
• Önceki Konuşma: {history}
• Doküman Bilgisi: {context1}  
• İstatistik Analizi: {context2}
• Rehberlik Kategorisi: {guidance_category}
• Rehberlik Template: {guidance_template}

SORU: {question}

YANITLAMA STRATEJİSİ:

1. REHBERLİK MODU KONTROLÜ:
   EĞER guidance_category boş değilse:
   - Template'deki sokratik yaklaşımı benimse
   - Kullanıcıyı keşfetmeye yönlendiren sorular sor
   - Direktif verme, rehberlik et
   - Kişiyi kendi tercihlerini keşfetmeye teşvik et
   - Template'i temel al ama doğal dilde ifade et

2. NORMAL MOD (guidance_category boş ise):
   - AKILLI BAĞLAM KULLANIMI: Önceki konuşma mevcut soruyla DOĞRUDAN alakalıysa dahil et
   - Farklı konu/soru türüyse önceki konuşmayı YOKSAY
   - Zoraki bağlantı kurma, doğal ve odaklanmış yanıt ver
   - ÖNCEKİ KONUŞMADA BELİRTİLEN BİLGİLERİ KULLAN (ilgi alanları, sıralama, tercihler)

3. KİŞİSELLEŞTİRME KURALLARI:
   - Kullanıcının belirttiği ilgi alanlarını yanıta entegre et
   - Sıralama bilgisi varsa spesifik önerilerde bulun
   - Genel tavsiye verme, kullanıcının durumuna özel yanıt ver
   - Tekrar etme, her yanıtta yeni değer kat

4. SORU TİPİNİ BELİRLE:
   - Genel rehberlik sorusu mu?
   - Spesifik veri/istatistik sorusu mu?
   - Önceki konuşmayla ilişkili mi?

5. KAYNAK SEÇİMİ:
   - Senin birikimin kaynaklarımızdan daha geniş, eğer Context1 veya Context2'de doğrudan soruya yanıt olabilecek bir bilgi yoksa kendi bilginden (veya contextlerden destek alarak) yanıt verebilirsin. 
   - Genel sorular: Kendi bilgin + Context1
   - İstatistik sorular: Context2 + Context1 + Kendi bilgin 
   - Önceki konuşma varsa: gerekliyse bağlamı dikkate al (daha çok son konuşmalar)

6. KAYNAK BELİRTME: 
   - SADECE CSV verilerinden rakam/oran/istatistik paylaşırken:
     "2024 Cumhurbaşkanlığı Uni-Veri Veritabanında yer alan bilgiye göre..."
   - Diğer tüm durumlarda kaynak belirtme

YANIT KURALLARI:
• REHBERLİK MODUNDA: Template'e sadık kal, sokratik sorular sor, kullanıcıyı yönlendir
• NORMAL MODDA: 3-4 cümle, net ve objektif, KİŞİSELLEŞTİRİLMİŞ (maksimum 400 kelime)
• FORMATLAMA KURALLARI (ÇOK ÖNEMLİ): 
  - Her madde arasında mutlaka boş bir satır bırak
  - Numaralı liste örneği:
    1. Birinci madde
    
    2. İkinci madde
    
    3. Üçüncü madde
  - Sorular sorarken alt alta yaz:
    👉 İlk soru?
    
    👉 İkinci soru?
  - Paragraflar arasında boş satır kullan
• TEKRAR ETME: Aynı bilgileri tekrar verme, her yanıtta yeni değer kat
• KİŞİSELLEŞTİR: Kullanıcının belirttiği ilgi alanları, sıralama gibi bilgileri kullan
• Önceki konuşmaya uygun ton SADECE alakalıysa
• Context2'yi sadece istatistik sorularında kullan
• Kendi vereceğin yanıt Context1'deki içerikten yanıta daha uygunsa kendi bilginle hareket edebilirsin.
• Kullanıcı dostu dil, teknik terimler yok
• Güncel bilgi (2020 sonrası)
• Kullanıcıyı kaynak dokümanlarımıza yönlendirme sadece kendi bilgini zenginleştirecek noktada Context1 ve Context2 yi kullan.
• Alakasız geçmişi zorlama, mevcut soruya odaklan

Yanıt:

ÖNEMLİ: Aşağıdaki format örneğindeki gibi maddeler arasında boş satır bırakarak yanıt ver. MUTLAKA tam cevap ver, yarıda kesme:

ÖRNEK FORMAT:
"Sıralamana uygun bölümler şunlar:

1. İlk bölüm hakkında açıklama

2. İkinci bölüm hakkında açıklama

3. Üçüncü bölüm hakkında açıklama

👉 İlk soru?

👉 İkinci soru?"

Bu formatı kullanarak EKSIKSIZ yanıt ver:"""

# CSV Tetikleyici Kelimeler - Değişiklik yok
CSV_KEYWORDS = [
    # Temel İstatistik Sorular
    "istihdam oranı", "çalışma oranı", "iş bulma", "mezun istihdamı",
    "maaş", "gelir", "kazanç", "ücret", "para kazanma",
    "sektör", "hangi sektör", "çalışma alanı", "iş sahası",
    "firma", "şirket", "işyeri", "çalıştığı yer",
    "işe başlama", "mezun olduktan sonra", "iş bulma süresi",
    "girişimcilik", "kendi işi", "startup", "girişim",
    
    # Spesifik Metrik Sorular  
    "yüzde kaç", "oranı nedir", "ne kadar", "hangi oranda",
    "istatistik", "veri", "sayısal", "rakam",
    "karşılaştır", "hangi bölüm daha", "en yüksek", "en düşük",
    
    # Maaş Aralıkları
    "17000", "25000", "34000", "51000", "maaş aralığı",
    "düşük maaş", "yüksek maaş", "ortalama maaş",
    
    # Zamanlama
    "kaç ayda", "ne kadar sürede", "hemen", "mezun olmadan",
    "6 ay", "12 ay", "1 yıl", "2 yıl"
]
