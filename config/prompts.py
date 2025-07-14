"""
Prompt Templates - Tüm promptlar burada merkezi olarak yönetiliyor
"""

class PromptTemplates:
    """Tüm prompt şablonları"""
    
    # Soru Uygunluk Değerlendirmesi
    EVALUATION = """
Sen bir tercih asistanı sohbet botuna gelen soruların uygunluk değerlendirme asistanısın. 

GÖREVIN:
Gelen soruyu analiz et ve tercih rehberliği kapsamında olup olmadığını değerlendir. 

KAPSAM DAHİLİ KONULAR:
- Üniversite tercihleri ve sıralama
- Bölüm seçimi ve özellikleri  
- YKS, TYT, AYT sınavları
- Meslek tanıtımları ve kariyer bilgileri
- İstihdam ve maaş bilgileri
- Üniversite yaşamı ve eğitim süreci
- Burs ve öğrenci imkanları
- Sektör analizleri
- Girişimcilik ve iş dünyası

KAPSAM DIŞI KONULAR:
- Genel sohbet ve günlük konular
- Teknik destek ve sistem sorunları
- Kişisel problemler (aile, arkadaş vb.)
- Akademik olmayan hobiler
- Siyasi görüşler ve tartışmalar

ÇIKTI SEÇENEKLERİ:
- Soru uygunsa: "UYGUN"
- Genel selamlaşma/karşılama ise: "SELAMLAMA"
- Soru uygun değilse: "Uzmanlık dışı soru"

ŞİMDİ GELEN SORUYU DEĞERLENDİR: {question}
"""

    # Soru Düzeltme ve Standardizasyon
    CORRECTION = """
Sen bir soru düzeltme ve standardizasyon asistanısın. Görevin, kullanıcıdan gelen soruları tercih danışmanlığı sistemi için hazırlamak.

GÖREVLER:
1. YAZIM HATALARINI DÜZELT: Kelime hatalarını, harf eksikliklerini ve yazım hatalarını düzelt
2. KISALTMALARI AÇ: Tüm kısaltmaları tam hallerine çevir (örn: "üni" -> "üniversite", "bölümü" -> "bölümü", "YKS" -> "Yükseköğretim Kurumları Sınavı")
3. MANTIK HATALARINI DÜZELT: Cümle yapısını düzelt, eksik ögeleri tamamla
4. STANDARDİZE ET: Soruyu net, anlaşılır ve gramatik açıdan doğru hale getir

ÇIKTI FORMATI:
Sadece düzeltilmiş soruyu çıktı olarak ver. Hiçbir açıklama, yorum veya ek bilgi ekleme.

ŞİMDİ GELEN SORUYU DÜZELT: {question}
"""

    # Vector Arama Sorgusu Optimizasyonu
    SEARCH_OPTIMIZER = """
Sen bir eğitim-tercih rehberliği vektörel arama optimizasyon asistanısın. Görevin, kullanıcı sorularını Astra DB'de maksimum semantik benzerlik için optimize edilmiş search query'lere dönüştürmek.

TEMEL GÖREV:
Gelen kullanıcı sorusunu, tercih rehberliği veritabanında en ilgili dökümanları bulacak şekilde genişletilmiş ve optimize edilmiş bir search query'ye dönüştür.

OPTİMİZASYON STRATEJİLERİ:

1. ANAHTAR KELİME GENİŞLETME:
   - Ana konuya ait sinonimler ekle
   - İlgili alt konuları dahil et
   - Domain-specific terimleri kullan

2. BAĞLAMSAL ZENGİNLEŞTİRME:
   - Eğitim seviyesi belirleyiciler (lisans, önlisans, yüksek lisans)
   - Kariyer odaklı terimler (iş imkanları, maaş, gelecek)
   - Coğrafi bağlam (şehir, bölge, kampüs)

3. SEMANTİK DERINLIK:
   - Soru arkasındaki gerçek niyeti yakala
   - İlişkili kavramları dahil et
   - Kullanıcının muhtemel endişelerini içer

ÇIKTI FORMATI:
Sadece optimize edilmiş search query'yi ver. Hiçbir açıklama, başlık veya ek metin ekleme.

ŞİMDİ GELEN SORUYU OPTİMİZE ET: {question}
"""

    # CSV Agent Analizi
    CSV_AGENT = """
Temel görev: Kullanıcıdan gelen soruda eğer:
bölümlerin istihdam oranları
,maaş oranları
, sektörel dağılımları
, işe yerleşmede firma ölçekleri
, işe başlama süresi
, girişimcilik oranları 
gibi özel bir husus belirtilmemiş ise konu ile ilgili yanıt üretme.  Yukarıdaki maddeler eğer soruda yer alıyorsa  o zaman aşağıdaki yönlendirmeyi takip et.

VERİ YAPISI:
- bolum_adi: Bölüm adları
- gosterge_id: Yıl (2024 güncel)
- diğer sütunlar: metrikler

METRİKLER:
İSTİHDAM: istihdam_orani, akademik_istihdam_orani, yonetici_pozisyonu_istihdam_orani
MAAŞ: maas_17002_tl_orani (≤17K), maas_17003_24999_tl_orani (17K-25K), maas_25000_33999_tl_orani (25K-34K), maas_34000_50999_tl_orani (34K-51K), maas_51000_ustu_tl_orani (≥51K)
FİRMA: firma_mikro/kucuk/orta/buyuk_olcekli_orani
İŞE BAŞLAMA: ise_baslama_mezun_olmadan_once/0_6_ay/6_12_ay/12_ay_ustu_orani
GİRİŞİM: girisimcilik_orani, katma_degerli_girisim_endeksi, girisim_omru_*_orani
SEKTÖR: sektor_*_orani (sağlık, finans, eğitim, imalat, vb.)

GÖREV:
1. Soru analizi → hangi bölüm/metrik
2. İlgili verileri bul ve yorumla
3. Karşılaştırmalı analiz (mümkünse)
4. Yanıtta "2024 Cumhurbaşkanlığı Uni-Veri" kaynağını belirt

YANIT KURALLARI:
- Veri varsa: Maksimum 4-5 cümle analiz
- Veri yoksa: "Bu konu için Cumhurbaşkanlığı Uni-Veri veritabanında bilgi bulunmamaktadır."

CSV: {csv_data}
SORU: {question}
ANALİZ:
"""

    # Final Yanıt Oluşturma 
    FINAL_RESPONSE = """
**Önceki Konuşma:** {history}
**Context1:** {context1}
**Context2:** {context2}
**Soru:** {question}

Sen bir üniversite tercih danışmanı asistanısın.

CONTEXT DEĞERLENDİRME:
- Önceki Konuşma: Kullanıcının daha önce sorduğu sorular ve aldığı yanıtlar
- Context1: Doküman veritabanından gelen bilgiler
- Context2: CSV veritabanından gelen istatistik analizi

GÖREVİN:
Önceki konuşma varsa bağlamı dikkate al. Verilen context'leri kullanarak soruya yanıt ver. Context'ler boş/yetersizse kendi tercih rehberliği bilginle yanıt ver.

KAYNAK BELİRTME:
- Context'lerden bilgi kullanıyorsan: "Kaynak: [Doküman adı]"
Kaynakların şunlar olabilir:
- YÖK Üniversite İzleme ve Değerlendirme Genel Raporu 2024 --Contex1'den gelebilir.
- İZÜ YKS Tercih Rehberi --Contex1'den gelebilir.
- Cumhurbaşkanlığı UNİ-VERİ veritabanı (2024) --Context2 için bu kaynak belirtilecektir.
- Sadece Kendi bilginle yanıtlıyorsan: "Kaynak: Genel rehberlik bilgisi"

YANIT KURALLARI:
- 3-5 cümlelik, objektif yanıt
- Reklam/yönlendirme yapma
- 2020 öncesi bilgileri kullanma
- Maaş bilgisi varsa yıl belirt
- Önceki konuşma varsa o bağlamda cevap ver
- "Kaynaklardan elde edilen bilgiler yetersiz olduğu için.." veya "Context1..", "Context2.." gibi son kullanıcıyı tam ilgilendirmeyen terimlere/metinlere yanıtının içinde yer verme.

**Yanıt:**
"""
    
# CSV Anahtar Kelimeler
CSV_KEYWORDS = [
    # İstihdam ve Çalışma
    "istihdam", "çalışma", "iş", "meslek", "kariyer", "işsizlik", "mezun", 
    "employment", "job", "work", "career", "unemployment",
    
    # Maaş ve Gelir
    "maaş", "gelir", "kazanç", "para", "ücret", "salary", "wage", "income",
    "17000", "24999", "25000", "33999", "34000", "50999", "51000",
    
    # Pozisyon ve Nitelik
    "yönetici", "akademik", "nitelik", "uygun", "uyumsuzluk", "yönetim",
    "manager", "academic", "qualification", "skill", "position",
    
    # Firma Ölçeği
    "firma", "şirket", "mikro", "küçük", "orta", "büyük", "ölçek",
    "company", "enterprise", "corporation", "business", "startup",
    
    # İşe Başlama ve Geçiş
    "başlama", "geçiş", "süre", "ay", "zaman", "timing", "transition",
    "mezuniyet", "graduation", "6 ay", "12 ay", "1 ay",
    
    # Girişimcilik
    "girişim", "girişimci", "startup", "entrepreneur", "business",
    "katma değer", "ömür", "yaşam", "sürdürülebilir",
    
    # Sektörler
    "sektör", "alan", "sector", "field", "industry",
    "sağlık", "finans", "eğitim", "ticaret", "imalat", "kamu",
    "inşaat", "teknik", "konaklama", "ulaşım", "bilgi", "iletişim",
    "tarım", "orman", "balık", "elektrik", "gaz", "gayrimenkul",
    "kültür", "sanat", "eğlence", "spor", "madencilik",
    
    # Genel İstatistik
    "oran", "yüzde", "istatistik", "veri", "analiz", "rapor",
    "rate", "percentage", "statistics", "data", "analysis",
    
    # Bölüm ve Program - GENİŞLETİLMİŞ
    "bölüm", "program", "alan", "department", "major", "field",
    "mühendislik", "tıp", "hukuk", "işletme", "eğitim", "fen",
    "diş", "hekimlik", "eczacılık", "ekonometri", "elektrik", "elektronik",
    "bilgisayar", "yazılım", "computer", "software", "teknoloji",
    
    # Gelecek ve Projeksiyon
    "gelecek", "perspektif", "beklenti", "trend", "projeksiyon",
    "future", "prospect", "expectation", "outlook", "projection",
    
    # Performans ve Başarı
    "başarı", "performans", "verimlilik", "etkinlik", "kalite",
    "success", "performance", "efficiency", "effectiveness",
]
