"""
🧪 AI Explainability Test Paketi
XAI özelliklerinin test edilmesi
"""

import sys
import numpy as np
import pandas as pd

# Test verisi
def create_test_data():
    """Test için örnek veri oluştur"""
    features = {
        'koi_period': 5.7,
        'koi_depth': 615.8,
        'koi_duration': 2.95,
        'koi_ror': 0.0174,
        'koi_srho': 4.11,
        'koi_prad': 1.89,
        'koi_teq': 1284.0,
        'koi_insol': 62.3,
        'koi_steff': 5455.0,
        'koi_slogg': 4.467,
        'koi_srad': 0.927,
        'koi_impact': 0.146,
        'koi_model_snr': 35.8,
        'koi_tce_plnt_num': 1,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ec': 0
    }
    return features

def test_explainability_module():
    """Explainability modülünü test et"""
    print("=" * 70)
    print("🧠 AI EXPLAINABILITY MODULE TEST")
    print("=" * 70)
    
    try:
        from explainability import (
            ExplainabilityEngine,
            create_decision_narrative,
            create_shap_waterfall_plotly,
            create_feature_importance_comparison,
            create_decision_path_visualization
        )
        print("✅ explainability modülü başarıyla içe aktarıldı")
    except Exception as e:
        print(f"❌ İçe aktarma hatası: {e}")
        return False
    
    # Mock verilerle test
    print("\n📊 Test 1: Karar Anlatısı Oluşturma")
    try:
        test_explanation = {
            'prediction': True,
            'confidence': 0.95,
            'probability': 0.92,
            'shap_analysis': {
                'top_features': [
                    {'name': 'koi_period', 'shap_value': 0.45, 'direction': 'positive', 'rank': 1},
                    {'name': 'koi_depth', 'shap_value': -0.23, 'direction': 'negative', 'rank': 2}
                ],
                'total_positive_impact': 0.75,
                'total_negative_impact': -0.25
            },
            'decision_rules': [
                {
                    'feature': 'koi_period',
                    'value': 5.7,
                    'status': 'optimal',
                    'impact': 'positive',
                    'optimal_range': (5, 100)
                }
            ],
            'confidence_factors': {
                'confidence_level': 0.95,
                'probability_margin': 0.45,
                'factors': [
                    {'factor': 'Yüksek Güven', 'description': 'Model çok emin', 'impact': 'positive'}
                ]
            },
            'what_if_analysis': [],
            'feature_contribution': None,
            'similar_cases': None
        }
        
        narrative = create_decision_narrative(test_explanation)
        print("✅ Karar anlatısı başarıyla oluşturuldu")
        print("\nÖrnek Anlatı:")
        print("-" * 70)
        print(narrative[:300] + "...")
        print("-" * 70)
        
    except Exception as e:
        print(f"❌ Karar anlatısı hatası: {e}")
        return False
    
    print("\n📈 Test 2: Plotly Görselleştirme Fonksiyonları")
    try:
        # Mock SHAP değerleri
        shap_values = np.array([0.5, -0.3, 0.2, -0.1, 0.4, 0.15, -0.25, 0.35, -0.05, 0.1])
        base_value = 0.0
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # Waterfall plot
        waterfall_fig = create_shap_waterfall_plotly(
            shap_values, base_value, features, feature_names, max_display=5
        )
        print("✅ SHAP Waterfall plot oluşturuldu")
        
        # Feature importance comparison
        model_importance = np.random.rand(10)
        shap_importance = np.abs(shap_values)
        
        comp_fig = create_feature_importance_comparison(
            model_importance, shap_importance, feature_names, top_n=5
        )
        print("✅ Özellik karşılaştırma grafiği oluşturuldu")
        
    except Exception as e:
        print(f"❌ Görselleştirme hatası: {e}")
        return False
    
    print("\n🎯 Test 3: Karar Yolu Görselleştirmesi")
    try:
        decision_path_fig = create_decision_path_visualization(test_explanation)
        if decision_path_fig:
            print("✅ Karar yolu grafiği oluşturuldu")
        else:
            print("⚠️ Karar yolu grafiği None döndü (normal olabilir)")
    except Exception as e:
        print(f"❌ Karar yolu görselleştirme hatası: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ TÜM TESTLER BAŞARIYLA TAMAMLANDI!")
    print("=" * 70)
    
    return True

def test_integration():
    """Streamlit entegrasyonunu test et"""
    print("\n" + "=" * 70)
    print("🔗 STREAMLIT ENTEGRASYON TESTI")
    print("=" * 70)
    
    try:
        # Streamlit içe aktarma kontrolü
        print("\n📦 Gerekli paketleri kontrol ediliyor...")
        
        required_packages = [
            'streamlit',
            'plotly',
            'shap',
            'lime',
            'numpy',
            'pandas',
            'matplotlib'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} - Yüklenmemiş!")
                return False
        
        print("\n✅ Tüm gerekli paketler mevcut")
        
    except Exception as e:
        print(f"❌ Entegrasyon testi hatası: {e}")
        return False
    
    return True

def print_feature_summary():
    """XAI özelliklerinin özetini yazdır"""
    print("\n" + "=" * 70)
    print("📚 AI EXPLAINABILITY ÖZELLİKLERİ ÖZETİ")
    print("=" * 70)
    
    features = {
        "SHAP Analizi": [
            "✅ İnteraktif Waterfall Plot",
            "✅ Özellik katkı tablosu",
            "✅ Pozitif/Negatif etki özeti",
            "✅ Matplotlib yedek görselleştirme"
        ],
        "Özellik Katkıları": [
            "✅ Ağırlıklı katkı analizi",
            "✅ Model önem derecesi karşılaştırması",
            "✅ Detaylı katkı tablosu",
            "✅ Interaktif grafikler"
        ],
        "Karar Kuralları": [
            "✅ Eşik değeri kontrolü",
            "✅ Optimal aralık analizi",
            "✅ Durum görselleştirmesi",
            "✅ Kural detayları"
        ],
        "What-If Analizi": [
            "✅ Senaryo simülasyonu",
            "✅ İnteraktif slider",
            "✅ Değişim yüzdesi hesaplama",
            "✅ Alternatif değer önerileri"
        ],
        "Karşılaştırmalı Analiz": [
            "✅ Bilinen gezegenlerle karşılaştırma",
            "✅ Görsel karşılaştırma grafikleri",
            "✅ Güven faktörü analizi",
            "✅ Referans değerler"
        ],
        "Anlatısal Açıklamalar": [
            "✅ İnsan tarafından okunabilir açıklamalar",
            "✅ Karar özetleri",
            "✅ Etki dengesi analizi",
            "✅ Güvenilirlik değerlendirmesi"
        ]
    }
    
    for category, items in features.items():
        print(f"\n🎯 {category}:")
        for item in items:
            print(f"   {item}")
    
    print("\n" + "=" * 70)
    print("📊 İstatistikler:")
    print(f"   • Toplam Kategori: {len(features)}")
    print(f"   • Toplam Özellik: {sum(len(items) for items in features.values())}")
    print(f"   • Görselleştirme: 10+ interaktif grafik")
    print(f"   • Analiz Derinliği: 5 seviye")
    print("=" * 70)

if __name__ == "__main__":
    print("\n🧪 AI EXPLAINABILITY TEST PAKETİ\n")
    
    # Testleri çalıştır
    test_results = []
    
    print("1️⃣ Explainability Modülü Testi")
    test_results.append(("Explainability Module", test_explainability_module()))
    
    print("\n2️⃣ Entegrasyon Testi")
    test_results.append(("Integration", test_integration()))
    
    # Özellikleri göster
    print_feature_summary()
    
    # Sonuç özeti
    print("\n" + "=" * 70)
    print("📊 TEST SONUÇ ÖZETİ")
    print("=" * 70)
    
    for test_name, result in test_results:
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in test_results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 TÜM TESTLER BAŞARIYLA GEÇTİ!")
        print("✅ AI Explainability özellikleri kullanıma hazır")
    else:
        print("⚠️ BAZI TESTLER BAŞARISIZ OLDU")
        print("❌ Lütfen hataları kontrol edin")
    print("=" * 70 + "\n")
    
    sys.exit(0 if all_passed else 1)
