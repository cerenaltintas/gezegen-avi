"""
Tahmin Örneği
Bu script, eğitilmiş modeli kullanarak tahmin yapmayı gösterir
"""

import joblib
import pandas as pd
import numpy as np

def load_model():
    """Eğitilmiş modeli yükle"""
    model = joblib.load('exoplanet_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

def predict_exoplanet(features_dict, model, scaler, feature_names):
    """Ötegezegen tahmini yap"""
    # Özellikleri DataFrame'e çevir
    X = pd.DataFrame([features_dict])
    
    # Özellik mühendisliği
    if 'koi_prad' in X.columns and 'koi_srad' in X.columns:
        X['planet_star_ratio'] = X['koi_prad'] / (X['koi_srad'] * 109.2)
    
    if 'koi_depth' in X.columns and 'koi_model_snr' in X.columns:
        X['signal_quality'] = X['koi_depth'] * X['koi_model_snr']
    
    if 'koi_period' in X.columns:
        X['orbital_velocity'] = 1 / X['koi_period']
    
    fp_flags = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    available_fp_flags = [f for f in fp_flags if f in X.columns]
    if available_fp_flags:
        X['fp_total_score'] = X[available_fp_flags].sum(axis=1)
    
    if 'koi_duration' in X.columns and 'koi_period' in X.columns:
        X['transit_shape_factor'] = X['koi_duration'] / X['koi_period']
    
    # Eksik özellikleri 0 ile doldur
    for feature in feature_names:
        if feature not in X.columns:
            X[feature] = 0
    
    # Özellikleri doğru sırada al
    X = X[feature_names]
    
    # Sonsuz ve NaN değerleri temizle
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Ölçeklendir ve tahmin yap
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return {
        'is_exoplanet': bool(prediction),
        'probability_not_exoplanet': float(probability[0]),
        'probability_exoplanet': float(probability[1]),
        'confidence': float(max(probability))
    }

# Örnek kullanım
if __name__ == "__main__":
    print("\n🔮 Ötegezegen Tahmin Örneği\n")
    
    # Modeli yükle
    print("Model yükleniyor...")
    model, scaler, feature_names = load_model()
    print("✅ Model yüklendi!\n")
    
    # Örnek 1: Kepler-452b benzeri gezegen (yaşanabilir bölgede)
    print("=" * 60)
    print("Örnek 1: Kepler-452b Benzeri Gezegen")
    print("=" * 60)
    
    example1 = {
        'koi_period': 384.8,          # Yörünge periyodu (gün)
        'koi_depth': 100.0,            # Geçiş derinliği (ppm)
        'koi_duration': 10.5,          # Geçiş süresi (saat)
        'koi_prad': 1.6,               # Gezegen yarıçapı (Dünya)
        'koi_ror': 0.012,              # Yarıçap oranı
        'koi_teq': 265,                # Denge sıcaklığı (K)
        'koi_insol': 1.1,              # Güneş ışınımı
        'koi_steff': 5757,             # Yıldız sıcaklığı (K)
        'koi_slogg': 4.32,             # Yıldız yerçekimi
        'koi_srad': 1.11,              # Yıldız yarıçapı
        'koi_srho': 1.4,               # Yıldız yoğunluğu
        'koi_impact': 0.5,             # Etki parametresi
        'koi_model_snr': 25.0,         # SNR
        'koi_tce_plnt_num': 1,         # Gezegen numarası
        'koi_fpflag_nt': 0,            # FP bayrakları
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ec': 0
    }
    
    result1 = predict_exoplanet(example1, model, scaler, feature_names)
    print(f"Sonuç: {'✅ ÖTEGEZEGEN' if result1['is_exoplanet'] else '❌ ÖTEGEZEGEN DEĞİL'}")
    print(f"Ötegezegen olma olasılığı: {result1['probability_exoplanet']*100:.2f}%")
    print(f"Güven skoru: {result1['confidence']*100:.2f}%\n")
    
    # Örnek 2: Sıcak Jüpiter
    print("=" * 60)
    print("Örnek 2: Sıcak Jüpiter Tipi Gezegen")
    print("=" * 60)
    
    example2 = {
        'koi_period': 3.52,            # Kısa yörünge periyodu
        'koi_depth': 5000.0,           # Yüksek geçiş derinliği
        'koi_duration': 3.2,           # Geçiş süresi
        'koi_prad': 11.2,              # Büyük gezegen (Jüpiter boyutu)
        'koi_ror': 0.11,               # Büyük yarıçap oranı
        'koi_teq': 1500,               # Çok sıcak
        'koi_insol': 850,              # Yüksek ışınım
        'koi_steff': 6200,             # Sıcak yıldız
        'koi_slogg': 4.4,
        'koi_srad': 1.3,
        'koi_srho': 0.8,
        'koi_impact': 0.3,
        'koi_model_snr': 45.0,         # Yüksek SNR
        'koi_tce_plnt_num': 1,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ec': 0
    }
    
    result2 = predict_exoplanet(example2, model, scaler, feature_names)
    print(f"Sonuç: {'✅ ÖTEGEZEGEN' if result2['is_exoplanet'] else '❌ ÖTEGEZEGEN DEĞİL'}")
    print(f"Ötegezegen olma olasılığı: {result2['probability_exoplanet']*100:.2f}%")
    print(f"Güven skoru: {result2['confidence']*100:.2f}%\n")
    
    # Örnek 3: Yanlış pozitif (yıldız tutulması)
    print("=" * 60)
    print("Örnek 3: Yanlış Pozitif (Yıldız Tutulması)")
    print("=" * 60)
    
    example3 = {
        'koi_period': 7.8,
        'koi_depth': 15000.0,          # Çok yüksek derinlik
        'koi_duration': 5.5,
        'koi_prad': 8.5,
        'koi_ror': 0.15,               # Çok büyük oran
        'koi_teq': 1200,
        'koi_insol': 400,
        'koi_steff': 5800,
        'koi_slogg': 4.3,
        'koi_srad': 1.1,
        'koi_srho': 1.2,
        'koi_impact': 0.8,
        'koi_model_snr': 30.0,
        'koi_tce_plnt_num': 1,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 1,            # Yıldız tutulması bayrağı
        'koi_fpflag_co': 1,            # Merkez sapması bayrağı
        'koi_fpflag_ec': 0
    }
    
    result3 = predict_exoplanet(example3, model, scaler, feature_names)
    print(f"Sonuç: {'✅ ÖTEGEZEGEN' if result3['is_exoplanet'] else '❌ ÖTEGEZEGEN DEĞİL'}")
    print(f"Ötegezegen olma olasılığı: {result3['probability_exoplanet']*100:.2f}%")
    print(f"Güven skoru: {result3['confidence']*100:.2f}%\n")
    
    print("=" * 60)
    print("✨ Tahmin örnekleri tamamlandı!")
    print("=" * 60)
