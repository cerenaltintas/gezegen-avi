"""
Ötegezegen Tespit Sistemi
Gerçek zamanlı NASA verileriyle desteklenen hibrit keşif modeli
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_generator import ExoplanetDataGenerator
from nasa_api import get_latest_dataframe

class ExoplanetDetector:
    """Ötegezegen tespit modeli sınıfı"""
    
    def __init__(self, data_path):
        """
        Parametreler:
            data_path: CSV veri dosyasının yolu
        """
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.df = None
        self.anomaly_detector = None
        
    def load_and_explore_data(self, df_override=None, source_name=None):
        """Veriyi yükle ve keşfet"""
        print("=" * 80)
        title = "VERİ YÜKLEME VE KEŞİF"
        if source_name:
            title = f"{title} · {source_name}"
        print(title)
        print("=" * 80)

        if df_override is not None:
            self.df = df_override.copy()
        else:
            # CSV dosyasını yükle (yorum satırlarını atla)
            self.df = pd.read_csv(self.data_path, comment='#')
        
        print(f"\n📊 Toplam veri noktası: {len(self.df)}")
        print(f"📊 Toplam özellik sayısı: {len(self.df.columns)}")
        
        # Hedef değişkeni kontrol et
        if 'koi_disposition' in self.df.columns:
            print(f"\n🎯 Hedef değişken dağılımı:")
            print(self.df['koi_disposition'].value_counts())
            print(f"\nYüzdeler:")
            print(self.df['koi_disposition'].value_counts(normalize=True) * 100)
        
        # Eksik veri analizi
        print(f"\n❌ Eksik veri oranları (>50% olanlar):")
        missing = (self.df.isnull().sum() / len(self.df)) * 100
        high_missing = missing[missing > 50].sort_values(ascending=False)
        if len(high_missing) > 0:
            print(high_missing)
        else:
            print("50%'den fazla eksik veriye sahip sütun yok")
        
        return self.df

    def load_live_data(self, mission='kepler', limit=500, force_refresh=False):
        """NASA Exoplanet Archive'dan canlı veri çek"""
        print("=" * 80)
        print(f"NASA EXOPLANET API ({mission.upper()})")
        print("=" * 80)

        live_df = get_latest_dataframe(mission=mission, limit=limit, force_refresh=force_refresh)
        print(f"\n📡 {mission.upper()} kataloğundan {len(live_df)} kayıt alındı")
        print(f"📅 Son güncelleme için {limit} kayıt sınırı kullanıldı")

        self.df = live_df
        return self.df
    
    def preprocess_data(self):
        """Veriyi ön işle ve özellik mühendisliği yap"""
        print("\n" + "=" * 80)
        print("VERİ ÖN İŞLEME VE ÖZELLİK MÜHENDİSLİĞİ")
        print("=" * 80)
        
        # Hedef değişkeni oluştur (CONFIRMED = 1, diğerleri = 0)
        self.df['is_exoplanet'] = (self.df['koi_disposition'] == 'CONFIRMED').astype(int)
        
        # En önemli özellikleri seç
        important_features = [
            'koi_period',           # Yörünge periyodu
            'koi_depth',            # Geçiş derinliği
            'koi_duration',         # Geçiş süresi
            'koi_ror',              # Gezegen-yıldız yarıçap oranı
            'koi_srho',             # Yıldız yoğunluğu
            'koi_prad',             # Gezegen yarıçapı
            'koi_teq',              # Denge sıcaklığı
            'koi_insol',            # Güneş ışınımı
            'koi_steff',            # Yıldız sıcaklığı
            'koi_slogg',            # Yıldız yüzey yerçekimi
            'koi_srad',             # Yıldız yarıçapı
            'koi_impact',           # Etki parametresi
            'koi_model_snr',        # Sinyal-gürültü oranı
            'koi_tce_plnt_num',     # Gezegen numarası
            'koi_fpflag_nt',        # Yanlış pozitif bayrakları
            'koi_fpflag_ss',
            'koi_fpflag_co',
            'koi_fpflag_ec',
        ]
        
        # Mevcut özellikleri filtrele
        available_features = [f for f in important_features if f in self.df.columns]
        print(f"✅ Kullanılacak özellikler ({len(available_features)}):")
        for feat in available_features:
            print(f"  - {feat}")
        
        # Özellikleri ve hedef değişkeni ayır
        X = self.df[available_features].copy()
        y = self.df['is_exoplanet'].copy()
        
        # Eksik değerleri doldur
        print(f"\n🔧 Eksik değerler medyan ile doldruluyor...")
        X = X.fillna(X.median())
        
        # Sonsuz değerleri temizle
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Özellik mühendisliği
        print(f"\n⚙️ Yeni özellikler oluşturuluyor...")
        
        # 1. Gezegen-yıldız boyut oranı
        if 'koi_prad' in X.columns and 'koi_srad' in X.columns:
            X['planet_star_ratio'] = X['koi_prad'] / (X['koi_srad'] * 109.2)  # Güneş yarıçapı -> Dünya yarıçapı
        
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
        
        # Son temizlik
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_names = X.columns.tolist()
        print(f"\n✨ Toplam özellik sayısı: {len(self.feature_names)}")
        
        return X, y

    def generate_synthetic_dataset(self, X, y, n_samples=2000, strategy='hybrid', output_path='synthetic_exoplanets.csv'):
        """Sentetik veri üret ve CSV olarak kaydet"""
        print("\n" + "=" * 80)
        print("SENTETİK VERİ ÜRETİMİ")
        print("=" * 80)

        generator = ExoplanetDataGenerator(random_state=42)
        generator.fit(X, y)
        synthetic_df = generator.generate_dataset(
            n_samples=n_samples,
            strategy=strategy,
            include_labels=True
        )
        synthetic_df.to_csv(output_path, index=False)

        print(f"✅ {len(synthetic_df)} satırlık yeni veri üretildi ve '{output_path}' dosyasına kaydedildi")
        class_breakdown = synthetic_df['is_exoplanet'].value_counts()
        for label, count in class_breakdown.items():
            ratio = count / len(synthetic_df) * 100
            print(f"  - Sınıf {label}: {count} (%{ratio:.2f})")

        return synthetic_df
    
    def balance_dataset(self, X, y):
        """Dengesiz veri setini dengele (SMOTE + Undersampling)"""
        print("\n" + "=" * 80)
        print("VERİ DENGELEme")
        print("=" * 80)
        
        print(f"📊 Dengeleme öncesi dağılım:")
        print(f"  - Ötegezegen değil (0): {(y == 0).sum()}")
        print(f"  - Ötegezegen (1): {(y == 1).sum()}")
        print(f"  - Dengesizlik oranı: {(y == 0).sum() / (y == 1).sum():.2f}:1")
        
        # SMOTE + Random Undersampling kombinasyonu
        over_sampler = SMOTE(sampling_strategy=0.5, random_state=42)
        under_sampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        
        X_balanced, y_balanced = over_sampler.fit_resample(X, y)
        X_balanced, y_balanced = under_sampler.fit_resample(X_balanced, y_balanced)
        
        print(f"\n📊 Dengeleme sonrası dağılım:")
        print(f"  - Ötegezegen değil (0): {(y_balanced == 0).sum()}")
        print(f"  - Ötegezegen (1): {(y_balanced == 1).sum()}")
        print(f"  - Yeni dengesizlik oranı: {(y_balanced == 0).sum() / (y_balanced == 1).sum():.2f}:1")
        
        return X_balanced, y_balanced
    
    def train_model(self, X_train, y_train):
        """XGBoost modelini eğit"""
        print("\n" + "=" * 80)
        print("MODEL EĞİTİMİ")
        print("=" * 80)
        
        # Veriyi ölçeklendir
        print("🔧 Veriler ölçeklendiriliyor (Robust Scaler)...")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # XGBoost modeli - optimize edilmiş parametreler
        print("🤖 XGBoost modeli oluşturuluyor...")
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        print("🎓 Model eğitiliyor...")
        self.model.fit(
            X_train_scaled, 
            y_train,
            verbose=False
        )
        
        print("🛰 Anomali dedektörü hazırlanıyor (IsolationForest)...")
        self.anomaly_detector = IsolationForest(
            n_estimators=256,
            contamination=0.02,
            random_state=42,
            n_jobs=-1,
        )
        self.anomaly_detector.fit(X_train_scaled)

        print("✅ Model eğitimi tamamlandı!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Modeli değerlendir ve metrikleri göster"""
        print("\n" + "=" * 80)
        print("MODEL DEĞERLENDİRME")
        print("=" * 80)
        
        # Tahmin yap
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Temel metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📈 PERFORMANS METRİKLERİ:")
        print(f"  - Doğruluk (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  - Kesinlik (Precision): {precision:.4f} ({precision*100:.2f}%)")
        print(f"  - Duyarlılık (Recall): {recall:.4f} ({recall*100:.2f}%)")
        print(f"  - F1 Skoru: {f1:.4f}")
        print(f"  - ROC AUC: {roc_auc:.4f}")
        
        # Karmaşıklık matrisi
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 KARMAŞIKLIK MATRİSİ:")
        print(f"  - Doğru Negatif (TN): {cm[0,0]}")
        print(f"  - Yanlış Pozitif (FP): {cm[0,1]}")
        print(f"  - Yanlış Negatif (FN): {cm[1,0]}")
        print(f"  - Doğru Pozitif (TP): {cm[1,1]}")
        
        # Detaylı rapor
        print(f"\n📋 DETAYLI SINIFLANDIRMA RAPORU:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Ötegezegen Değil', 'Ötegezegen']))
        
        # Özellik önem dereceleri
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n⭐ EN ÖNEMLİ 10 ÖZELLİK:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        novelty_scores = None
        novelty_flags = None
        if self.anomaly_detector is not None:
            novelty_scores = self.anomaly_detector.decision_function(X_test_scaled)
            novelty_flags = self.anomaly_detector.predict(X_test_scaled)
            # IsolationForest returns -1 for anomaly
            novel_count = int((novelty_flags == -1).sum())
            print(f"\n🧭 Potansiyel yeni aday sayısı: {novel_count}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'novelty_scores': novelty_scores,
            'novelty_flags': novelty_flags
        }
    
    def save_model(self, model_path='exoplanet_model.pkl', scaler_path='scaler.pkl'):
        """Modeli ve scaler'ı kaydet"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, 'feature_names.pkl')
        if self.anomaly_detector is not None:
            joblib.dump(self.anomaly_detector, 'anomaly_detector.pkl')
        print(f"\n💾 Model kaydedildi: {model_path}")
        print(f"💾 Scaler kaydedildi: {scaler_path}")
        print(f"💾 Özellik isimleri kaydedildi: feature_names.pkl")
        if self.anomaly_detector is not None:
            print("💾 Anomali modeli kaydedildi: anomaly_detector.pkl")
    
    def load_model(self, model_path='exoplanet_model.pkl', scaler_path='scaler.pkl'):
        """Kaydedilmiş modeli yükle"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load('feature_names.pkl')
        anomaly_path = Path('anomaly_detector.pkl')
        if anomaly_path.exists():
            self.anomaly_detector = joblib.load(anomaly_path)
        print(f"✅ Model yüklendi: {model_path}")
    
    def predict_single(self, features_dict):
        """Tek bir veri noktası için tahmin yap"""
        # Özellikleri DataFrame'e çevir
        X = pd.DataFrame([features_dict])
        
        # Eksik özellikleri 0 ile doldur
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        # Özellikleri doğru sırada al
        X = X[self.feature_names]
        
        # Ölçeklendir ve tahmin yap
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        novelty = None
        if self.anomaly_detector is not None:
            novelty_flag = self.anomaly_detector.predict(X_scaled)[0]
            novelty_score = self.anomaly_detector.decision_function(X_scaled)[0]
            novelty = {
                'is_novel_candidate': bool(novelty_flag == -1),
                'novelty_score': float(novelty_score)
            }

        return {
            'is_exoplanet': bool(prediction),
            'probability_not_exoplanet': float(probability[0]),
            'probability_exoplanet': float(probability[1]),
            'confidence': float(max(probability)),
            'novelty': novelty
        }

    def detect_novel_candidates(self, X):
        """Veri kümesindeki potansiyel yeni adayları döndür"""
        if self.anomaly_detector is None:
            raise RuntimeError("Anomali modeli yüklenmeden bu fonksiyon çağrılamaz")

        X_local = X.copy()
        missing_cols = [col for col in self.feature_names if col not in X_local.columns]
        for col in missing_cols:
            X_local[col] = 0
        X_local = X_local[self.feature_names]

        X_scaled = self.scaler.transform(X_local)
        novelty_flags = self.anomaly_detector.predict(X_scaled)
        novelty_scores = self.anomaly_detector.decision_function(X_scaled)

        result = X_local.copy()
        result['novelty_score'] = novelty_scores
        result['is_novel_candidate'] = novelty_flags == -1
        return result[result['is_novel_candidate']]
    
    def plot_results(self, metrics):
        """Sonuçları görselleştir"""
        print("\n📊 Grafikler oluşturuluyor...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Karmaşıklık Matrisi
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Karmaşıklık Matrisi')
        axes[0, 0].set_xlabel('Tahmin')
        axes[0, 0].set_ylabel('Gerçek')
        axes[0, 0].set_xticklabels(['Değil', 'Ötegezegen'])
        axes[0, 0].set_yticklabels(['Değil', 'Ötegezegen'])
        
        # 2. Metrik Karşılaştırması
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        metrics_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['roc_auc']
        ]
        axes[0, 1].bar(metrics_names, metrics_values, color='skyblue')
        axes[0, 1].set_title('Model Performans Metrikleri')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='%90 Hedef')
        for i, v in enumerate(metrics_values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        axes[0, 1].legend()
        
        # 3. Özellik Önem Dereceleri (Top 15)
        top_features = metrics['feature_importance'].head(15)
        axes[1, 0].barh(top_features['feature'], top_features['importance'], color='coral')
        axes[1, 0].set_title('En Önemli 15 Özellik')
        axes[1, 0].set_xlabel('Önem Derecesi')
        axes[1, 0].invert_yaxis()
        
        # 4. Özellik Önem Dağılımı
        axes[1, 1].hist(metrics['feature_importance']['importance'], bins=30, 
                       color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('Özellik Önem Derecesi Dağılımı')
        axes[1, 1].set_xlabel('Önem Derecesi')
        axes[1, 1].set_ylabel('Frekans')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        print("✅ Grafikler 'model_performance.png' olarak kaydedildi")
        plt.show()


def main():
    """Ana program"""
    parser = argparse.ArgumentParser(
        description="Ötegezegen tespit modelini eğit ve değerlendir"
    )
    parser.add_argument(
        "--source",
        choices=["local", "live", "hybrid"],
        default="local",
        help="Veri kaynağını seç (yerel CSV, canlı NASA verisi veya hibrit)"
    )
    parser.add_argument(
        "--mission",
        choices=["kepler"],
        default="kepler",
        help="Canlı veri için NASA görevi (yalnızca Kepler desteklenir)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="NASA API'dan çekilecek maksimum kayıt sayısı"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="NASA API önbelleğini zorla yenile"
    )
    parser.add_argument(
        "--data-path",
        default='cumulative_2025.10.04_09.55.40.csv',
        help="Yerel Kepler CSV veri dosyasının yolu"
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Sentetik veri üretimini atla"
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=3000,
        help="Üretilecek sentetik örnek sayısı"
    )
    parser.add_argument(
        "--synthetic-strategy",
        choices=["hybrid", "positive", "negative", "physical"],
        default="hybrid",
        help="Sentetik veri üretim stratejisi"
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🌌 ÖTEGEZEGEN KEŞFETME SİSTEMİ 🌌")
    print("NASA Kepler Veri Seti ile Makine Öğrenimi")
    print("=" * 80 + "\n")

    # Veri dosyası yolu
    data_path = args.data_path

    # Detector'ı oluştur
    detector = ExoplanetDetector(data_path)

    # 1. Veriyi yükle ve keşfet
    if args.source == "local":
        df = detector.load_and_explore_data(source_name="Yerel CSV")
    elif args.source == "live":
        live_df = detector.load_live_data(
            mission=args.mission,
            limit=args.limit,
            force_refresh=args.force_refresh
        )
        df = detector.load_and_explore_data(
            df_override=live_df,
            source_name=f"NASA {args.mission.upper()} canlı verisi"
        )
    else:
        # Hibrit veri seti: yerel + NASA canlı verisi
        print("🌐 Hibrit veri hazırlığı başlatılıyor...")
        live_df = detector.load_live_data(
            mission=args.mission,
            limit=args.limit,
            force_refresh=args.force_refresh
        )
        local_df = pd.read_csv(data_path, comment='#')
        combined_df = pd.concat([local_df, live_df], ignore_index=True, sort=False)
        dedupe_keys = [col for col in ['kepid', 'koi_name', 'tic_id', 'planet_name'] if col in combined_df.columns]
        if dedupe_keys:
            combined_df = combined_df.drop_duplicates(subset=dedupe_keys, keep='last')
        else:
            combined_df = combined_df.drop_duplicates()
        combined_df = combined_df.reset_index(drop=True)
        print(f"🧬 Hibrit veri seti oluşturuldu: {len(local_df)} yerel + {len(live_df)} canlı kayıt")
        df = detector.load_and_explore_data(
            df_override=combined_df,
            source_name=f"Hibrit ({args.mission.upper()} + yerel)"
        )
    
    # 2. Veriyi ön işle
    X, y = detector.preprocess_data()

    # 2.1 Sentetik veri üret
    synthetic_df = None
    if args.skip_synthetic or args.synthetic_samples <= 0:
        print("⏭ Sentetik veri üretimi kullanıcı tercihiyle atlandı.")
    else:
        synthetic_df = detector.generate_synthetic_dataset(
            X,
            y,
            n_samples=args.synthetic_samples,
            strategy=args.synthetic_strategy
        )
    
    # 3. Train-test split
    print("\n" + "=" * 80)
    print("VERİ BÖLÜNMESİ")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Eğitim seti: {len(X_train)} örnek")
    print(f"✅ Test seti: {len(X_test)} örnek")
    
    # 4. Veriyi dengele
    X_train_balanced, y_train_balanced = detector.balance_dataset(X_train, y_train)
    
    # 5. Modeli eğit
    detector.train_model(X_train_balanced, y_train_balanced)
    
    # 6. Modeli değerlendir
    metrics = detector.evaluate_model(X_test, y_test)
    
    # 7. Modeli kaydet
    detector.save_model()
    
    # 8. Sonuçları görselleştir
    detector.plot_results(metrics)
    
    # 9. Örnek tahmin
    print("\n" + "=" * 80)
    print("ÖRNEK TAHMİN")
    print("=" * 80)
    print("\n🔮 Test setinden rastgele bir örnek seçiliyor...")
    
    sample_idx = np.random.randint(0, len(X_test))
    sample_features = X_test.iloc[sample_idx].to_dict()
    true_label = y_test.iloc[sample_idx]
    
    prediction = detector.predict_single(sample_features)
    
    print(f"\n📊 Gerçek Durum: {'✅ ÖTEGEZEGEN' if true_label == 1 else '❌ ÖTEGEZEGEN DEĞİL'}")
    print(f"🤖 Model Tahmini: {'✅ ÖTEGEZEGEN' if prediction['is_exoplanet'] else '❌ ÖTEGEZEGEN DEĞİL'}")
    print(f"📈 Ötegezegen Olma Olasılığı: {prediction['probability_exoplanet']*100:.2f}%")
    print(f"📈 Güven Skoru: {prediction['confidence']*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("✨ Program tamamlandı! ✨")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
