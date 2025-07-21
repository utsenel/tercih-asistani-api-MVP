"""
İyileştirilmiş Prompt Templates - Basitleştirilmiş Kaynak Sistemi
"""

class PromptTemplates:
    """Optimize edilmiş prompt şablonları"""
    
       # YENİ: Birleştirilmiş Smart Evaluator-Corrector
    SMART_EVALUATOR_CORRECTOR = """
GÖREV: Gelen soruyu eğer gerekliyse önceki konuşma bağlamıyla değerlendirip optimize et.
DİKKAT: Her soru geçmiş konuşmaya ihtiyaç duymayabilir.

GEÇMIŞ KONUŞMA:
{history}

GÜNCEL SORU: {question}

ADIM 1 - BAĞLAM ANALİZİ:
• Sorunun önceki konuşma ile zenginleşmesi gerekiyor mu ona karar ver.
• Önceki konuşmada spesifik bir bölüm/konu/meslek var mı?
• Güncel soru önceki konuşmayla ilişkili mi? ("peki", "o zaman", "bunun" gibi bağlayıcılar)
• Eksik referans var mı? ("onun maaşı", "bu bölümde", "orada" gibi)


ADIM 2 - UYGUNLUK DEĞERLENDİRMESİ:
KAPSAM DAHİLİ:
• Üniversite/bölüm tercihi, sıralama, karşılaştırma
• YKS/TYT/AYT sınavları, puan türleri
• Kariyer/meslek bilgisi, gelecek planları
• İstihdam/maaş verileri, iş imkanları
• Eğitim süreci/kampüs yaşamı
• Burs/öğrenci imkanları

KAPSAM DIŞI:
• Genel sohbet, gündelik konular
• Teknik sorunlar, sistem hataları
• Kişisel/aile meseleleri
• Siyasi görüşler, ideolojik konular

SELAMLAMA İNDİKATÖRLERİ:
• "merhaba", "selam", "iyi günler", "nasılsın"
• "yardım", "neler yapabilirsin", "kimsin"

ADIM 3 - SORU OPTİMİZASYONU:
• Bağlamsal bilgiyi soruya entegre et (bölüm adı, meslek, önceki konu)
• Yazım hatalarını düzelt, kısaltmaları aç
• Belirsizlikleri gider, eksik referansları tamamla
• Tercih rehberliği terminolojisini kullan

ÇIKTI FORMATI (kesinlikle bu formatta):
STATUS: [UYGUN/SELAMLAMA/KAPSAM_DIŞI]
ENHANCED_QUESTION: [Context-aware düzeltilmiş soru]

ÖRNEK:
Geçmiş: "user: bilgisayar mühendisliği nasıl bir bölüm?"
Güncel: "peki maaşları nasıl?"
STATUS: UYGUN
ENHANCED_QUESTION: Bilgisayar mühendisliği mezunlarının maaş durumu ve gelir seviyeleri nasıl?
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

    # Final Response - Basitleştirilmiş kaynak sistemi
    FINAL_RESPONSE = """
BAĞLAM:
• Önceki Konuşma: {history}
• Doküman Bilgisi: {context1}  
• İstatistik Analizi: {context2}

SORU: {question}

YANITLAMA STRATEJİSİ:

1. SORU TİPİNİ BELİRLE:
   - Genel rehberlik sorusu mu?
   - Spesifik veri/istatistik sorusu mu?
   - Önceki konuşmayla ilişkili mi?

2. KAYNAK SEÇİMİ:
   - Senin birikimin kaynaklarımızdan daha geniş, eğer Context1 veya Context2'de doğrudan soruya yanıt olabilecek bir bilgi yoksa kendi bilginden (veya contextlerden destek alarak) yanıt verebilirsin. 
   - Genel sorular: Kendi bilgin + Context1
   - İstatistik sorular: Context2 + Context1 + Kendi bilgin 
   - Önceki konuşma varsa: gerekliyse bağlamı dikkate al (daha çok son konuşmalar)

3. KAYNAK BELİRTME: 
   - SADECE CSV verilerinden rakam/oran/istatistik paylaşırken:
     "2024 Cumhurbaşkanlığı Uni-Veri Veritabanında yer alan bilgiye göre..."
   - Diğer tüm durumlarda kaynak belirtme

YANIT KURALLARI:
• 3-5 cümle, net ve objektif
• Önceki konuşmaya uygun ton
• Context2'yi sadece istatistik sorularında kullan
• Kendi vereceğin yanıt Context1'deki içerikten yanıta daha uygunsa kendi bilginle hareket edebilirsin.
• Kullanıcı dostu dil, teknik terimler yok
• Güncel bilgi (2020 sonrası)
• Kullanıcıyı kaynak dokümanlarımıza yönlendirme sadece kendi bilgini zenginleştirecek noktada Context1 ve Context2 yi kullan.
• Alakalı değilse historyden bahsetme.

Yanıt:"""

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
