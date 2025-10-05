"""
Canlı NASA verilerini test et - Model değerlendirmesi ile
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Model ve scaler'ı yükle
print("=" * 70)
print("MODEL DEĞERLENDİRME TESTİ")
print("=" * 70)

try:
    with open('exoplanet_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("✓ Model, scaler ve feature_names yüklendi")
except Exception as e:
    print(f"✗ Model dosyaları yüklenemedi: {e}")
    sys.exit(1)

# NASA API'den veri al
try:
    from nasa_api import get_latest_dataframe
    from streamlit_app import score_catalog
    
    print("\n1️⃣ NASA API'den canlı veri alınıyor...")
    df = get_latest_dataframe(mission="kepler", limit=100, force_refresh=True)
    print(f"   ✓ {len(df)} kayıt alındı")
    
except Exception as e:
    print(f"   ✗ HATA: {e}")
    sys.exit(1)

# Modelle değerlendir
try:
    print("\n2️⃣ Model ile değerlendiriliyor...")
    scoring = score_catalog(df, model, scaler, list(feature_names), anomaly_detector=None)
    scored_df = scoring["scored"]
    
    print(f"   ✓ {len(scored_df)} obje skorlandı")
    
    # İstatistikler
    if "model_prediction" in scored_df.columns and "model_probability" in scored_df.columns:
        predicted_exoplanets = scored_df[scored_df["model_prediction"] == "CONFIRMED"]
        predicted_non_exoplanets = scored_df[scored_df["model_prediction"] == "OTHER"]
        high_confidence = scored_df[scored_df["model_probability"] >= 0.9]
        medium_confidence = scored_df[scored_df["model_probability"].between(0.7, 0.9)]
        low_confidence = scored_df[scored_df["model_probability"] < 0.7]
        
        print(f"\n3️⃣ MODEL TAHMİN SONUÇLARI:")
        print(f"   🪐 Ötegezegen Tahmini: {len(predicted_exoplanets):,}")
        print(f"   ❌ Ötegezegen Değil: {len(predicted_non_exoplanets):,}")
        print(f"   🎯 Yüksek Güven (≥90%): {len(high_confidence):,}")
        print(f"   ⚡ Orta Güven (70-90%): {len(medium_confidence):,}")
        print(f"   ⚠️ Düşük Güven (<70%): {len(low_confidence):,}")
        
        # NASA etiketleri ile karşılaştırma
        if "koi_disposition" in scored_df.columns:
            nasa_confirmed = scored_df[scored_df["koi_disposition"].str.upper() == "CONFIRMED"]
            nasa_candidate = scored_df[scored_df["koi_disposition"].str.upper() == "CANDIDATE"]
            
            print(f"\n4️⃣ NASA vs MODEL KARŞILAŞTIRMASI:")
            print(f"   ✅ NASA CONFIRMED: {len(nasa_confirmed):,}")
            
            if len(nasa_confirmed) > 0:
                model_agrees = nasa_confirmed[nasa_confirmed["model_prediction"] == "CONFIRMED"]
                agreement_rate = len(model_agrees) / len(nasa_confirmed) * 100
                print(f"   🎯 Model Uyumu: %{agreement_rate:.1f} ({len(model_agrees)}/{len(nasa_confirmed)})")
            
            if len(nasa_candidate) > 0:
                model_promotes = nasa_candidate[nasa_candidate["model_prediction"] == "CONFIRMED"]
                print(f"   🔍 NASA CANDIDATE: {len(nasa_candidate):,}")
                print(f"   ⬆️ Model Terfi: {len(model_promotes):,} objeyi CONFIRMED olarak değerlendirdi")
        
        # En yüksek güvenle tahmin edilenler
        if len(predicted_exoplanets) > 0:
            print(f"\n5️⃣ EN GÜVEN İLİR ÖTEGEZEGEN TAHMİNLERİ (Top 5):")
            top_5 = predicted_exoplanets.nlargest(5, "model_probability")
            
            for idx, row in top_5.iterrows():
                name = row.get("kepoi_name", row.get("koi_name", f"Index {idx}"))
                prob = row["model_probability"]
                nasa_disp = row.get("koi_disposition", "N/A")
                period = row.get("koi_period", "N/A")
                prad = row.get("koi_prad", "N/A")
                
                print(f"   • {name}")
                print(f"     Model Güveni: %{prob*100:.1f}")
                print(f"     NASA Etiketi: {nasa_disp}")
                print(f"     Periyot: {period:.2f} gün" if isinstance(period, (int, float)) else f"     Periyot: {period}")
                print(f"     Yarıçap: {prad:.2f} R⊕" if isinstance(prad, (int, float)) else f"     Yarıçap: {prad}")
                print()
    
    print("=" * 70)
    print("✅ TEST BAŞARILI!")
    print("=" * 70)
    
except Exception as e:
    print(f"   ✗ HATA: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
