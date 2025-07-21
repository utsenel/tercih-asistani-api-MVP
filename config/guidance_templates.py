"""
Rehberlik Template Library - Döküman örneklerinden distile edilmiş
"""

class GuidanceTemplates:
    """
    Belirsiz/kararsız adaylar için sokratik yaklaşım template'leri
    """
    
    # Ana kategoriler ve template'leri
    TEMPLATES = {
        "GENEL_BELIRSIZLIK": {
            "description": "Ne okuyayım, kafam karışık, bilmiyorum türü sorular",
            "approach": "Seni tanıma soruları, keşif odaklı",
            "template": """Önce seni tanımama yardımcı olur musun? 😊

👉 Hangi konular ilgini çeker?
👉 Lisede hangi derslerde daha keyif aldın?
👉 Hayalini kurduğun bir iş ortamı var mı?

Bunlardan başlayalım mı? Sen anlatırken birlikte yolumuzu çizeriz."""
        },
        
        "SIRALAMA_ENDISESI": {
            "description": "Kötü mü yaptım, yeter mi, gelir mi türü endişeler",
            "approach": "Sıralamayı relativize et, hedefe odakla",
            "template": """Sıralama sadece bir araç, asıl önemli olan hangi bölümü neden istediğin. 😊

👉 Hedefin neydi, hangi alanı istiyorsun?
👉 Sen hangi işleri yaparken mutlu olursun?

Önce hedeflerinden başlayalım, sonra sıralamana uygun seçeneklere birlikte bakabiliriz."""
        },
        
        "BOLUM_KARSILASTIRMA": {
            "description": "X mi Y mi daha iyi, hangisi garanti türü sorular",
            "approach": "Beceri setine odakla, garantici olmama",
            "template": """Artık "hangi bölüm daha iyi" sorusu yerine hangi beceriyi geliştirmek istediğin önemli.

👉 Sen hangi tür işlerde kendini daha mutlu hissedersin?
👉 Teknik problemler mi yoksa insanlarla çalışmak mı seni daha çok heyecanlandırır?

Bunu konuştuktan sonra hangi bölümün sana daha uygun olduğunu birlikte değerlendirebiliriz."""
        },
        
        "GARANTI_ARAYISI": {
            "description": "En iyi bölüm, garanti iş, işsiz kalmam türü sorular",
            "approach": "Gerçekçi iş dünyası perspektifi, beceri vurgusu",
            "template": """Günümüzde "garanti iş" sadece bölüm seçimine bağlı değil - hangi beceriyi geliştirdiğin daha önemli.

👉 Sen hangi alanda kendini geliştirmekten keyif alırsın?
👉 Çalışırken seni motive eden şey ne olur?

Bu soruları konuştuktan sonra hem ilgini çeken hem de gelecek vadeden alanları birlikte değerlendirebiliriz."""
        },
        
        "SEHIR_KARARSIZLIGI": {
            "description": "İstanbul mı Ankara mı, büyük şehir mi küçük şehir mi",
            "approach": "Kişisel tercihleri keşfetme",
            "template": """Şehir seçimi üniversite deneyimini ciddi şekilde etkiler.

👉 Kalabalık ve hızlı tempolu ortamları mı seversin yoksa daha sakin yerler mi?
👉 Yaşam maliyeti mi daha önemli yoksa sosyal imkanlar mı?
👉 Aile desteğine ne kadar ihtiyaç duyarsın?

Bu tercihleri konuştuktan sonra hangi şehirlerin sana daha uygun olacağını değerlendirebiliriz."""
        },
        
        "VAKIF_DEVLET_IKILEMI": {
            "description": "Vakıf mı devlet mi, hangisi daha iyi",
            "approach": "Kriterleri netleştirme",
            "template": """Vakıf ya da devlet üniversitesi seçimi senin önceliklerine bağlı.

👉 Burs olanakları senin için ne kadar önemli?
👉 Küçük sınıflar mı yoksa geniş kampüs imkanları mı tercih edersin?
👉 Yaşam maliyeti bütçen nasıl?

Bu kriterleri konuştuktan sonra hangi seçeneğin sana daha uygun olduğunu birlikte değerlendirebiliriz."""
        },
        
        "MESLEK_SEKTOR_MERAK": {
            "description": "Hangi meslek, ne iş yapar, sektör merakı",
            "approach": "Ilgi alanı keşfi odaklı",
            "template": """Meslek seçimi için önce hangi tür işlerin seni heyecanlandırdığını anlamamız gerekir.

👉 Daha çok ekip çalışması mı yoksa bireysel çalışma mı seversin?
👉 Ofis ortamı mı, saha çalışması mı tercih edersin?
👉 Yaratıcı işler mi yoksa analitik işler mi ilgini çeker?

Bu tercihlerin doğrultusunda sana uygun meslek alanlarını birlikte keşfedebiliriz."""
        },
        
        "META_BOT": {
            "description": "Sen kimsin, nasıl çalışıyorsun türü meta sorular",
            "approach": "Kendini tanıtma, rol açıklama",
            "template": """Ben bir üniversite tercih rehberliği asistanıyım! 🎓

**Nasıl çalışıyorum:**
• Senin ilgi alanlarını, yeteneklerini ve hedeflerini anlamaya çalışırım
• YKS tercihleri, bölüm seçimi, kariyer planlaması konularında yardımcı olurum
• Sana hazır cevap vermek yerine, doğru soruları sorarak düşünmeni kolaylaştırırım

**Ne konularda yardımcı olabilirim:**
👉 Bölüm seçimi ve karşılaştırma
👉 Üniversite/şehir tercihi
👉 Kariyer planlama
👉 İstihdam ve maaş verileri
👉 Tercih stratejileri

Sen de bana hangi konuda yardıma ihtiyaç duyduğunu söyleyebilirsin! 😊"""
        }
    }
    
    # Kategori tespiti için anahtar kelimeler
    CATEGORY_KEYWORDS = {
        "GENEL_BELIRSIZLIK": [
            "ne okuyayım", "bilmiyorum", "kafam karışık", "karar veremiyorum",
            "hiçbir şey istemiyorum", "ne yapmak istediğimi bilmiyorum",
            "hangi bölüm", "ne seçeyim", "öneriniz", "yardım edin",
            "hiçbir bölümü sevmiyorum", "ne yapmalıyım"
        ],
        
        "SIRALAMA_ENDISESI": [
            "kötü mü", "iyi mi", "yeter mi", "gelir mi", "sıralama",
            "bin", "puan", "başarısız", "düşük", "yüksek",
            "geçer mi", "alır mı", "tutturabilir miyim", "yaptım"
        ],
        
        "BOLUM_KARSILASTIRMA": [
            " mi ", " mı ", "hangisi", "karşılaştır", "arasında",
            "daha iyi", "daha avantajlı", "tercih", "seçim"
        ],
        
        "GARANTI_ARAYISI": [
            "garanti", "en iyi", "işsiz kalmam", "iş bulur", "güvenli",
            "kesin", "mutlaka", "garantili", "işsizlik", "iş imkanı",
            "hangi bölüm işsiz kalmaz", "en çok iş"
        ],
        
        "SEHIR_KARARSIZLIGI": [
            "istanbul", "ankara", "izmir", "şehir", "nerede okuyayım",
            "büyük şehir", "küçük şehir", "yaşam", "konaklama"
        ],
        
        "VAKIF_DEVLET_IKILEMI": [
            "vakıf", "devlet", "özel", "ücret", "burs", "para",
            "maliyet", "hangisi daha iyi"
        ],
        
        "MESLEK_SEKTOR_MERAK": [
            "ne iş", "hangi meslek", "çalışma alanı", "sektör",
            "iş yapar", "görev", "sorumluluk", "kariyer"
        ],
        
        "META_BOT": [
            "sen kimsin", "nasıl çalışıyorsun", "neler yapabilirsin",
            "insanla mı konuşuyorum", "robot musun", "yapay zeka mısın",
            "bana nasıl yardımcı olacaksın", "ne tür sorular sorabilirim",
            "kim olduğunu", "ne yapabildiğini", "hangi konularda yardımcı"
        ]
    }
    
    @classmethod
    def detect_category(cls, question: str) -> str:
        """
        Sorudan rehberlik kategorisini tespit et
        """
        question_lower = question.lower()
        
        # Her kategori için keyword kontrolü
        category_scores = {}
        
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in question_lower:
                    # Uzun keyword'lere daha yüksek puan
                    score += len(keyword.split())
            
            if score > 0:
                category_scores[category] = score
        
        # En yüksek skorlu kategoriyi döndür
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return ""
    
    @classmethod
    def get_template(cls, category: str) -> str:
        """
        Kategori için template döndür
        """
        if category in cls.TEMPLATES:
            return cls.TEMPLATES[category]["template"]
        return ""
    
    @classmethod
    def get_all_categories(cls) -> list:
        """
        Tüm kategorileri listele
        """
        return list(cls.TEMPLATES.keys())

# Test fonksiyonu
def test_category_detection():
    """
    Kategori tespitini test et
    """
    test_questions = [
        "Ne okuyayım bilmiyorum",
        "300 bin sıralamayla iyi bir bölüm gelir mi?",
        "Bilgisayar mühendisliği mi endüstri mühendisliği mi daha iyi?",
        "Hangi bölüm garanti iş bulur?",
        "İstanbul'da mı okumalıyım Ankara'da mı?",
        "Vakıf üniversitesi mi devlet mi daha iyi?"
    ]
    
    for question in test_questions:
        category = GuidanceTemplates.detect_category(question)
        print(f"Soru: {question}")
        print(f"Kategori: {category}")
        print(f"Template: {GuidanceTemplates.get_template(category)[:50]}...")
        print("-" * 50)

if __name__ == "__main__":
    test_category_detection()
