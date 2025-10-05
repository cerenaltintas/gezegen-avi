"""
🤖 Gemini AI Entegrasyonu
Google Gemini API ile XAI açıklamalarını zenginleştirme
"""

import os
import json
from typing import Dict, Optional
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')


class GeminiExplainer:
    """
    Google Gemini API ile AI destekli açıklamalar
    """
    
    def __init__(self, api_key: str):
        """
        Args:
            api_key: Google Gemini API anahtarı
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        # Gemini 2.0 Flash - hızlı ve etkili model
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def generate_explanation(self, prediction_data: Dict, xai_data: Dict) -> Optional[str]:
        """
        Tahmin ve XAI verileri için Gemini ile açıklama oluştur
        
        Args:
            prediction_data: Tahmin sonuçları
            xai_data: XAI analiz verileri
            
        Returns:
            str: Gemini tarafından oluşturulan açıklama veya None
        """
        try:
            # Prompt oluştur
            prompt = self._create_prompt(prediction_data, xai_data)
            
            # API isteği
            response = self._call_gemini_api(prompt)
            
            return response
            
        except Exception as e:
            print(f"Gemini API hatası: {e}")
            return None
    
    def _create_prompt(self, prediction_data: Dict, xai_data: Dict) -> str:
        """
        Gemini için SHAP odaklı detaylı prompt oluştur
        """
        # Tahmin bilgileri
        is_exoplanet = prediction_data.get('is_exoplanet', False)
        confidence = prediction_data.get('confidence', 0) * 100
        probability = prediction_data.get('probability_exoplanet', 0) * 100
        
        # SHAP bilgileri - DETAYLı
        shap_info = ""
        shap_total_impact = 0
        if xai_data.get('shap_analysis'):
            shap = xai_data['shap_analysis']
            top_features = shap.get('top_features', [])[:5]
            
            shap_info = "\n**🎯 SHAP DEĞERLERİ (Model Kararındaki Gerçek Etkiler):**\n"
            for i, feat in enumerate(top_features, 1):
                direction = "ARTTIRDI ⬆️" if feat['direction'] == 'positive' else "AZALTTI ⬇️"
                impact_percent = abs(feat['shap_value']) * 100
                shap_total_impact += abs(feat['shap_value'])
                shap_info += f"{i}. **{feat['name']}**: SHAP = {feat['shap_value']:.4f} → Ötegezegen olasılığını %{impact_percent:.1f} {direction}\n"
            
            # Toplam etki
            pos_impact = shap.get('total_positive_impact', 0)
            neg_impact = shap.get('total_negative_impact', 0)
            shap_info += f"\n📊 **Toplam SHAP Etkisi:**\n"
            shap_info += f"   - Pozitif etkiler toplamı: +{pos_impact:.3f}\n"
            shap_info += f"   - Negatif etkiler toplamı: {neg_impact:.3f}\n"
            shap_info += f"   - Net etki: {pos_impact + neg_impact:.3f}\n"
        
        # Özellik katkıları
        contrib_info = ""
        if xai_data.get('feature_contributions'):
            contrib = xai_data['feature_contributions']
            contrib_info = "\n**📈 ÖZELLİK DEĞERLERİ:**\n"
            for item in contrib.get('top_contributors', [])[:3]:
                contrib_info += f"- {item['feature']}: {item['value']:.3f} (Katkı: {item['contribution']:.3f})\n"
        
        # Karar kuralları
        rules_info = ""
        if xai_data.get('decision_rules'):
            rules = xai_data['decision_rules']
            rules_info = "\n**✅ KRİTİK EŞİK KONTROLÜ:**\n"
            for rule in rules[:3]:
                status_map = {
                    'optimal': '✅ Optimal Aralıkta',
                    'acceptable': '⚠️ Kabul Edilebilir',
                    'out_of_range': '❌ Normal Dışı'
                }
                status = status_map.get(rule['status'], '❓')
                rules_info += f"- {rule['feature']}: {rule['value']:.3f} {status}\n"
        
        # Ana prompt - SHAP ODAKLI
        prompt = f"""
Sen bir XAI (Explainable AI) uzmanısın. NASA Kepler ötegezegen tespit modelinin kararını **SHAP (SHapley Additive exPlanations) değerlerine dayanarak** açıklayacaksın.

🔬 **ÖNEMLİ:** SHAP değerleri, her özelliğin model kararına **matematiksel olarak tam katkısını** gösterir. Senin görevin bu SHAP değerlerini yorumlamak.

**TAHMIN SONUCU:**
- Karar: {"ÖTEGEZEGEN 🪐" if is_exoplanet else "ÖTEGEZEGEN DEĞİL ❌"}
- Model Güveni: %{confidence:.2f}
- Ötegezegen Olasılığı: %{probability:.2f}

{shap_info}

{contrib_info}

{rules_info}

**GÖREV - SHAP ODAKLI AÇIKLAMA:**

1. **SHAP Analizi (en önemli bölüm):**
   - En yüksek SHAP değerine sahip ilk 3 özelliği belirt
   - Her özelliğin SHAP değerinin ne anlama geldiğini açıkla
   - Pozitif/negatif SHAP değerlerinin toplamını yorumla
   - Hangi özelliklerin kararı "ötegezegen" yönüne, hangilerinin "değil" yönüne ittiğini açıkla

2. **Güvenilirlik Değerlendirmesi:**
   - SHAP değerlerinin dağılımına göre (tek bir özellik mi dominant, yoksa birçok özellik mi katkı veriyor?) güvenilirliği yorumla
   - Çelişkili SHAP değerleri var mı? (bazı özellikler artırırken bazıları azaltıyor mu?)

3. **Sonuç:**
   - SHAP değerlerine dayalı nihai yorum
   - Bu karar ne kadar sağlam? (SHAP değerlerinin tutarlılığına göre)

**FORMAT:**
- Markdown kullan, emojiler ekle
- 200-300 kelime
- **SHAP değerlerini esas al**, genel yorum yapma
- Türkçe yaz

**YASAK:**
❌ Genel fizik bilgisi ekleme
❌ SHAP değerlerinde olmayan şeyler hakkında konuşma
❌ "Muhtemelen", "sanırım" gibi belirsiz ifadeler
✅ Sadece verilen SHAP değerlerini yorumla
"""
        
        return prompt
    
    def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """
        Gemini API'ye istek gönder
        """
        try:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            print(f"Gemini API hatası: {e}")
            return None
    
    def generate_short_summary(self, prediction_data: Dict) -> Optional[str]:
        """
        Kısa özet için Gemini kullan
        
        Args:
            prediction_data: Tahmin sonuçları
            
        Returns:
            str: Kısa özet (1-2 cümle)
        """
        try:
            is_exoplanet = prediction_data.get('is_exoplanet', False)
            confidence = prediction_data.get('confidence', 0) * 100
            
            prompt = f"""
Bir ötegezegen tespit modeli şu sonucu verdi:
- Karar: {"Ötegezegen" if is_exoplanet else "Ötegezegen Değil"}
- Güven: %{confidence:.1f}

Bu sonucu 1-2 cümlede özetle. Kısa ve net ol. Türkçe yaz.
"""
            
            response = self._call_gemini_api(prompt)
            return response
            
        except Exception as e:
            print(f"Özet oluşturma hatası: {e}")
            return None
    
    def explain_feature_importance(self, feature_name: str, shap_value: float, 
                                   feature_value: float, is_exoplanet: bool) -> Optional[str]:
        """
        Belirli bir özelliğin önemini açıkla
        
        Args:
            feature_name: Özellik adı
            shap_value: SHAP değeri
            feature_value: Özellik değeri
            is_exoplanet: Ötegezegen mi?
            
        Returns:
            str: Özellik açıklaması
        """
        try:
            direction = "artırdı" if shap_value > 0 else "azalttı"
            
            prompt = f"""
Ötegezegen tespit modelinde '{feature_name}' özelliği, tahmine {abs(shap_value):.3f} değerinde etki etti ve ötegezegen olasılığını {direction}.

Özelliğin değeri: {feature_value:.3f}
Model tahmini: {"Ötegezegen" if is_exoplanet else "Ötegezegen Değil"}

Bu özelliğin neden bu kadar önemli olduğunu ve değerinin ne anlama geldiğini 2-3 cümlede açıkla. 
Bilimsel terimler kullan ama anlaşılır ol. Türkçe yaz.
"""
            
            response = self._call_gemini_api(prompt)
            return response
            
        except Exception as e:
            print(f"Özellik açıklama hatası: {e}")
            return None


def test_gemini_integration(api_key: str):
    """
    Gemini entegrasyonunu test et
    """
    print("=" * 70)
    print("🤖 GEMINI AI ENTEGRASYON TESTİ")
    print("=" * 70)
    
    # GeminiExplainer oluştur
    try:
        explainer = GeminiExplainer(api_key)
        print("✅ GeminiExplainer başarıyla oluşturuldu")
    except Exception as e:
        print(f"❌ GeminiExplainer oluşturulamadı: {e}")
        return False
    
    # Test verisi
    test_prediction = {
        'is_exoplanet': True,
        'confidence': 0.95,
        'probability_exoplanet': 0.92,
        'probability_not_exoplanet': 0.08
    }
    
    test_xai = {
        'shap_analysis': {
            'top_features': [
                {'name': 'koi_period', 'shap_value': 0.45, 'direction': 'positive', 'rank': 1},
                {'name': 'koi_depth', 'shap_value': -0.23, 'direction': 'negative', 'rank': 2},
                {'name': 'koi_model_snr', 'shap_value': 0.35, 'direction': 'positive', 'rank': 3}
            ],
            'total_positive_impact': 0.85,
            'total_negative_impact': -0.23
        },
        'decision_rules': [
            {'feature': 'koi_period', 'value': 5.7, 'status': 'optimal', 'impact': 'positive'},
            {'feature': 'koi_prad', 'value': 1.89, 'status': 'optimal', 'impact': 'positive'}
        ],
        'confidence_factors': {
            'confidence_level': 0.95,
            'probability_margin': 0.42,
            'factors': [
                {'factor': 'Yüksek Güven', 'description': 'Model çok emin', 'impact': 'positive'}
            ]
        }
    }
    
    # Test 1: Ana açıklama
    print("\n📊 Test 1: Ana Açıklama Üretimi")
    print("Gemini API'ye istek gönderiliyor...")
    
    explanation = explainer.generate_explanation(test_prediction, test_xai)
    
    if explanation:
        print("✅ Açıklama başarıyla oluşturuldu\n")
        print("-" * 70)
        print(explanation)
        print("-" * 70)
    else:
        print("❌ Açıklama oluşturulamadı")
        return False
    
    # Test 2: Kısa özet
    print("\n📝 Test 2: Kısa Özet Üretimi")
    summary = explainer.generate_short_summary(test_prediction)
    
    if summary:
        print("✅ Özet başarıyla oluşturuldu")
        print(f"Özet: {summary}")
    else:
        print("⚠️ Özet oluşturulamadı (opsiyonel)")
    
    print("\n" + "=" * 70)
    print("✅ GEMINI ENTEGRASYONU BAŞARILI!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    import sys
    
    # API anahtarını ortam değişkeninden veya argümandan al
    api_key = os.getenv('GEMINI_API_KEY')
    
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    if not api_key:
        print("❌ GEMINI_API_KEY bulunamadı!")
        print("Kullanım: python gemini_explainer.py YOUR_API_KEY")
        sys.exit(1)
    
    # Testi çalıştır
    success = test_gemini_integration(api_key)
    sys.exit(0 if success else 1)
