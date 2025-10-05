"""
Ötegezegen Keşif Sistemi Web Arayüzü
Flask ile interaktif kullanıcı arayüzü
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Dosya yükleme konfigürasyonu
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Global değişkenler
model = None
scaler = None
feature_names = None

def allowed_file(filename):
    """Dosya uzantısının geçerli olup olmadığını kontrol et"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def anonymize_sensitive_columns(df, sensitive_columns=None):
    """
    Veri setindeki hassas sütunları anonimleştirir.
    
    Args:
        df: Pandas DataFrame
        sensitive_columns: Anonimleştirilecek sütun listesi
    
    Returns:
        Anonimleştirilmiş DataFrame ve rapor
    """
    df_anon = df.copy()
    anonymization_report = {
        'total_columns': len(df.columns),
        'anonymized_columns': [],
        'method_used': {}
    }
    
    # Otomatik hassas sütun tespiti
    if sensitive_columns is None:
        sensitive_columns = []
        sensitive_keywords = ['name', 'id', 'email', 'address', 'phone', 
                             'ra', 'dec', 'coordinate', 'location', 
                             'kepid', 'kepoi']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in sensitive_keywords):
                sensitive_columns.append(col)
    
    # Anonimleştirme
    for col in sensitive_columns:
        if col not in df.columns:
            continue
            
        col_type = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            df_anon[col] = df[col].apply(
                lambda x: int(hashlib.sha256(str(x).encode()).hexdigest()[:8], 16) % 1000000
                if pd.notna(x) else x
            )
            anonymization_report['method_used'][col] = 'numeric_hash'
        
        elif pd.api.types.is_string_dtype(col_type) or col_type == 'object':
            df_anon[col] = df[col].apply(
                lambda x: f"ANON_{hashlib.sha256(str(x).encode()).hexdigest()[:12].upper()}"
                if pd.notna(x) else x
            )
            anonymization_report['method_used'][col] = 'string_hash'
        
        anonymization_report['anonymized_columns'].append(col)
    
    return df_anon, anonymization_report

def load_model_files():
    """Model dosyalarını yükle"""
    global model, scaler, feature_names
    
    try:
        model = joblib.load('exoplanet_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        print("✅ Model dosyaları başarıyla yüklendi!")
        return True
    except Exception as e:
        print(f"⚠️ Model dosyaları yüklenemedi: {e}")
        print("ℹ️ Lütfen önce main.py dosyasını çalıştırın.")
        return False

def predict_exoplanet(features_dict):
    """Ötegezegen tahmini yap"""
    try:
        # Özellikleri DataFrame'e çevir
        X = pd.DataFrame([features_dict])
        
        # Özellik mühendisliği (main.py ile aynı)
        # 1. Gezegen-yıldız boyut oranı
        if 'koi_prad' in X.columns and 'koi_srad' in X.columns:
            X['planet_star_ratio'] = X['koi_prad'] / (X['koi_srad'] * 109.2)
        
        # 2. Sinyal kalitesi göstergesi
        if 'koi_depth' in X.columns and 'koi_model_snr' in X.columns:
            X['signal_quality'] = X['koi_depth'] * X['koi_model_snr']
        
        # 3. Yörünge hızı tahmini
        if 'koi_period' in X.columns:
            X['orbital_velocity'] = 1 / X['koi_period']
        
        # 4. Yanlış pozitif toplam skoru
        fp_flags = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        available_fp_flags = [f for f in fp_flags if f in X.columns]
        if available_fp_flags:
            X['fp_total_score'] = X[available_fp_flags].sum(axis=1)
        
        # 5. Geçiş şekil faktörü
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
            'success': True,
            'is_exoplanet': bool(prediction),
            'probability_not_exoplanet': float(probability[0]),
            'probability_exoplanet': float(probability[1]),
            'confidence': float(max(probability)),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin endpoint'i"""
    try:
        # Form verilerini al
        features = {}
        
        # Temel özellikler
        features['koi_period'] = float(request.form.get('koi_period', 0))
        features['koi_depth'] = float(request.form.get('koi_depth', 0))
        features['koi_duration'] = float(request.form.get('koi_duration', 0))
        features['koi_ror'] = float(request.form.get('koi_ror', 0))
        features['koi_srho'] = float(request.form.get('koi_srho', 0))
        features['koi_prad'] = float(request.form.get('koi_prad', 0))
        features['koi_teq'] = float(request.form.get('koi_teq', 0))
        features['koi_insol'] = float(request.form.get('koi_insol', 0))
        features['koi_steff'] = float(request.form.get('koi_steff', 0))
        features['koi_slogg'] = float(request.form.get('koi_slogg', 0))
        features['koi_srad'] = float(request.form.get('koi_srad', 0))
        features['koi_impact'] = float(request.form.get('koi_impact', 0))
        features['koi_model_snr'] = float(request.form.get('koi_model_snr', 0))
        features['koi_tce_plnt_num'] = float(request.form.get('koi_tce_plnt_num', 1))
        
        # Yanlış pozitif bayrakları
        features['koi_fpflag_nt'] = int(request.form.get('koi_fpflag_nt', 0))
        features['koi_fpflag_ss'] = int(request.form.get('koi_fpflag_ss', 0))
        features['koi_fpflag_co'] = int(request.form.get('koi_fpflag_co', 0))
        features['koi_fpflag_ec'] = int(request.form.get('koi_fpflag_ec', 0))
        
        # Tahmin yap
        result = predict_exoplanet(features)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """CSV dosyası yükleyerek toplu tahmin"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Dosya bulunamadı'
            })
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Dosya seçilmedi'
            })
        
        # Dosya güvenliği kontrolü
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Geçersiz dosya türü. Sadece CSV dosyaları kabul edilir. Dosya: {file.filename}'
            })
        
        # Dosya boyutu kontrolü
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'Dosya çok büyük ({file_size / (1024*1024):.2f} MB). Maksimum {MAX_FILE_SIZE / (1024*1024):.0f} MB olmalıdır.'
            })
        
        # Güvenli dosya adı
        filename = secure_filename(file.filename)
        
        # CSV dosyasını oku
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            return jsonify({
                'success': False,
                'error': 'CSV dosyası boş'
            })
        except pd.errors.ParserError as e:
            return jsonify({
                'success': False,
                'error': f'CSV ayrıştırma hatası: {str(e)}'
            })
        
        # Veri anonimleştirme (opsiyonel, form parametresi ile kontrol edilebilir)
        anonymize = request.form.get('anonymize', 'true').lower() == 'true'
        anonymization_report = None
        
        if anonymize:
            df, anonymization_report = anonymize_sensitive_columns(df)
        
        # Her satır için tahmin yap
        results = []
        for idx, row in df.iterrows():
            features = row.to_dict()
            prediction = predict_exoplanet(features)
            
            if prediction['success']:
                results.append({
                    'row': idx + 1,
                    'is_exoplanet': prediction['is_exoplanet'],
                    'probability': prediction['probability_exoplanet'],
                    'confidence': prediction['confidence']
                })
        
        # İstatistikler
        total = len(results)
        exoplanets_found = sum(1 for r in results if r['is_exoplanet'])
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        response_data = {
            'success': True,
            'filename': filename,
            'total_rows': total,
            'exoplanets_found': exoplanets_found,
            'non_exoplanets': total - exoplanets_found,
            'average_confidence': float(avg_confidence),
            'results': results[:100]  # İlk 100 sonucu döndür
        }
        
        # Anonimleştirme bilgisi ekle
        if anonymization_report:
            response_data['anonymization'] = {
                'enabled': True,
                'columns_anonymized': len(anonymization_report['anonymized_columns']),
                'column_names': anonymization_report['anonymized_columns']
            }
        else:
            response_data['anonymization'] = {'enabled': False}
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model_info')
def model_info():
    """Model bilgileri"""
    try:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return jsonify({
            'success': True,
            'model_type': 'XGBoost Classifier',
            'total_features': len(feature_names),
            'top_features': [
                {'name': row['feature'], 'importance': float(row['importance'])}
                for _, row in feature_importance.head(10).iterrows()
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🌌 ÖTEGEZEGEN KEŞİF SİSTEMİ - WEB ARAYÜZÜ 🌌")
    print("=" * 80 + "\n")
    
    # Model dosyalarını yükle
    if load_model_files():
        print("\n🚀 Web sunucusu başlatılıyor...")
        print("📡 Adres: http://127.0.0.1:5000")
        print("💡 Tarayıcınızda yukarıdaki adresi açın\n")
        print("=" * 80 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Model dosyaları bulunamadı!")
        print("ℹ️ Lütfen önce 'python main.py' komutunu çalıştırın.\n")
