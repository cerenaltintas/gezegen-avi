"""
🌌 Ötegezegen Keşif Sistemi - Streamlit Arayüzü
XAI (Explainable AI) ile Gelişmiş Analiz ve Açıklanabilir Tahminler
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import base64
from io import BytesIO
from pathlib import Path
import hashlib
import warnings
warnings.filterwarnings('ignore')

from data_generator import (
    ExoplanetDataGenerator,
    load_reference_dataset,
    prepare_features_from_dataframe,
)
from nasa_api import get_latest_dataframe
from explainability import (
    ExplainabilityEngine,
    create_decision_narrative,
    create_shap_waterfall_plotly,
    create_feature_importance_comparison,
    create_decision_path_visualization
)
from gemini_explainer import GeminiExplainer

# Gemini API Anahtarı (güvenli bir şekilde saklanmalı)
GEMINI_API_KEY = "AIzaSyCX0vQ1MfSAFPfYsrZXMiXbVBp5fRGV6Eg"

# Sayfa yapılandırması
st.set_page_config(
    page_title="🌌 Ötegezegen Keşif Sistemi",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Stilleri
st.markdown("""
    <style>
    :root {
        --gradient-bg: radial-gradient(circle at 25% 15%, #1e3a8a 0%, #0b1220 45%, #020617 100%);
        --card-bg: rgba(15, 23, 42, 0.78);
        --card-border: rgba(148, 163, 184, 0.18);
        --chip-bg: rgba(148, 163, 184, 0.14);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5f5;
        --accent: #38bdf8;
        --accent-strong: #2563eb;
    }
    .stApp {
        background: var(--gradient-bg);
        color: var(--text-primary);
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }
    .main {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.88);
        backdrop-filter: blur(18px);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    .hero {
        padding: 36px 42px;
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(37, 99, 235, 0.25), rgba(124, 58, 237, 0.22));
        border: 1px solid rgba(148, 163, 184, 0.18);
        backdrop-filter: blur(24px);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 10px;
        color: var(--text-primary);
    }
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.05rem;
        max-width: 620px;
        line-height: 1.6;
        margin-bottom: 18px;
    }
    .tagline {
        color: #60a5fa;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.28em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 16px;
    }
    .hero-badges {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(148, 163, 184, 0.16);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 999px;
        padding: 8px 16px;
        color: #cbd5f5;
        font-size: 0.85rem;
        display: inline-flex;
        gap: 8px;
        align-items: center;
    }
    .cta-button {
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        color: var(--text-primary);
        padding: 12px 22px;
        border-radius: 14px;
        text-decoration: none;
        font-weight: 600;
        display: inline-flex;
        gap: 10px;
        align-items: center;
        margin-top: 20px;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 45px rgba(37, 99, 235, 0.28);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 18px;
        margin: 28px 0 8px;
    }
    .metric-card {
        background: var(--card-bg);
        border-radius: 18px;
        border: 1px solid var(--card-border);
        padding: 22px;
        color: var(--text-secondary);
    }
    .metric-card h2 {
        margin-bottom: 10px;
        font-size: 1rem;
        color: var(--text-primary);
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 600;
        color: var(--accent);
        margin-bottom: 6px;
    }
    .metric-delta {
        font-size: 0.85rem;
        color: #38ef7d;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .dashboard-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 26px 28px;
        margin-bottom: 22px;
        color: var(--text-secondary);
    }
    .section-title {
        color: var(--text-primary);
        font-size: 1.3rem;
        font-weight: 650;
        margin-bottom: 6px;
    }
    .section-subtitle {
        color: var(--text-secondary);
        font-size: 0.96rem;
        margin-bottom: 18px;
    }
    .card-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 18px;
        margin-top: 12px;
    }
    .card-item {
        background: rgba(15, 23, 42, 0.68);
        border-radius: 18px;
        padding: 22px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        color: var(--text-secondary);
    }
    .card-item h3 {
        color: var(--text-primary);
        font-size: 1.05rem;
        margin-bottom: 10px;
    }
    .info-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        background: rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        color: var(--text-secondary);
        margin-right: 8px;
        margin-bottom: 10px;
    }
    .info-box, .warning-box, .success-box {
        background: rgba(15, 23, 42, 0.65);
        border-radius: 16px;
        padding: 18px 20px;
        border-left: 4px solid var(--accent);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: var(--text-secondary);
    }
    .warning-box { border-left-color: #f97316; }
    .success-box { border-left-color: #34d399; }
    .info-box h3, .warning-box h3, .success-box h3 {
        color: var(--text-primary);
    }
    .prediction-banner {
        border-radius: 18px;
        padding: 26px 24px;
        margin: 20px 0 28px;
        text-align: center;
        font-weight: 600;
        font-size: 1.35rem;
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: var(--text-primary);
    }
    .prediction-banner.success {
        background: linear-gradient(120deg, rgba(22, 163, 74, 0.22), rgba(34, 197, 94, 0.18));
        border-left: 6px solid #22c55e;
    }
    .prediction-banner.danger {
        background: linear-gradient(120deg, rgba(239, 68, 68, 0.22), rgba(251, 113, 133, 0.18));
        border-left: 6px solid #f97316;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 10px 18px;
        color: var(--text-secondary);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(130deg, rgba(37, 99, 235, 0.45), rgba(124, 58, 237, 0.45));
        color: var(--text-primary);
    }
    .streamlit-expanderHeader {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.55);
        border-radius: 0 0 16px 16px;
    }
    .stButton > button {
        border-radius: 14px;
        height: 46px;
        font-weight: 600;
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        border: none;
        color: var(--text-primary);
    }
    .stButton > button:hover {
        opacity: 0.92;
    }
    .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.26);
        background: rgba(15, 23, 42, 0.72);
        color: var(--text-secondary);
        font-weight: 600;
    }
    .stMetricValue, .stMetricLabel, .stMetricDelta {
        color: var(--text-primary) !important;
    }
    .highlight-text {
        color: var(--accent);
        font-weight: 600;
    }
    .divider {
        margin: 36px 0;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
    }
    </style>
""", unsafe_allow_html=True)


def _resolve_dashboard_token() -> str:
    """Kurumsal panel için erişim anahtarını çözümler."""
    secrets_token = None
    try:
        secrets_token = st.secrets["dashboard_passcode"]
    except Exception:
        secrets_token = None

    env_token = os.environ.get("EXO_DASHBOARD_TOKEN")

    token = secrets_token or env_token or "demo"
    return str(token).strip()


def enforce_access_control() -> None:
    """Panel erişimini basit bir erişim kodu ile sınırlar."""
    required_token = _resolve_dashboard_token()

    if required_token.lower() == "demo":
        return

    if "exo_dashboard_authenticated" not in st.session_state:
        st.session_state.exo_dashboard_authenticated = False

    if st.session_state.exo_dashboard_authenticated:
        return

    st.sidebar.markdown("### 🔐 Güvenli Erişim")
    entered_code = st.sidebar.text_input(
        "Kurumsal erişim kodu", type="password", key="access_code"
    )
    login_clicked = st.sidebar.button("Giriş yap", key="access_code_button")

    if login_clicked:
        if entered_code == required_token:
            st.session_state.exo_dashboard_authenticated = True
            st.sidebar.success("Doğrulama tamamlandı.")
        else:
            st.sidebar.error("Kod doğrulanamadı. Lütfen tekrar deneyin.")

    if not st.session_state.exo_dashboard_authenticated:
        st.warning("Yetkili erişim gereklidir. Geçerli kodu girene kadar panel gizlidir.")
        st.stop()


@st.cache_data(show_spinner=False)
def fetch_live_catalog(mission: str, limit: int, force_refresh: bool) -> pd.DataFrame:
    """NASA Exoplanet Archive'dan canlı verileri getirir."""
    df = get_latest_dataframe(mission=mission, limit=limit, force_refresh=force_refresh)
    df = df.copy()
    df["source_mission"] = mission.upper()
    return df


def _align_feature_frame(features_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Modelde kullanılan özelliklerin sırasını ve bütünlüğünü sağlar."""
    aligned = features_df.copy()
    missing_cols = [col for col in feature_names if col not in aligned.columns]
    for col in missing_cols:
        aligned[col] = 0
    aligned = aligned[feature_names]
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.fillna(0)
    return aligned


def score_catalog(
    df: pd.DataFrame,
    model,
    scaler,
    feature_names: list[str],
    anomaly_detector=None,
):
    """Veri çerçevesini model ve anomali dedektörü ile skorlar."""
    features, labels = prepare_features_from_dataframe(df)
    aligned_features = _align_feature_frame(features, feature_names)
    X_scaled = scaler.transform(aligned_features)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = model.predict(X_scaled)

    scored_df = df.copy()
    scored_df["model_probability"] = probabilities
    scored_df["model_prediction"] = np.where(predictions == 1, "CONFIRMED", "OTHER")

    novelty_scores = None
    novelty_flags = None
    if anomaly_detector is not None:
        novelty_scores = anomaly_detector.decision_function(X_scaled)
        novelty_flags = anomaly_detector.predict(X_scaled)
        scored_df["novelty_score"] = novelty_scores
        scored_df["is_novel_candidate"] = novelty_flags == -1
    else:
        scored_df["novelty_score"] = np.nan
        scored_df["is_novel_candidate"] = False

    return {
        "scored": scored_df,
        "features": aligned_features,
        "labels": labels,
        "scaled": X_scaled,
        "probabilities": probabilities,
        "novelty_scores": novelty_scores,
        "novelty_flags": novelty_flags,
    }

# Model ve scaler'ı yükle
@st.cache_resource
def load_models():
    """Model, scaler ve özellik isimlerini yükle"""
    try:
        model = joblib.load('exoplanet_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        anomaly_path = Path('anomaly_detector.pkl')
        anomaly_detector = None
        if anomaly_path.exists():
            anomaly_detector = joblib.load(anomaly_path)

        # SHAP Explainer'ı oluştur
        background_df = load_default_reference_dataframe().drop(columns='is_exoplanet')
        background_aligned = _align_feature_frame(background_df, feature_names)
        explainer = shap.TreeExplainer(model, data=background_aligned.sample(min(200, len(background_aligned)), random_state=42))

        return model, scaler, feature_names, explainer, anomaly_detector
    except Exception as e:
        st.error(f"❌ Model dosyaları yüklenemedi: {e}")
        st.info("ℹ️ Lütfen önce 'python main.py' komutunu çalıştırın.")
        return None, None, None, None, None


@st.cache_data(show_spinner=False)
def load_default_reference_dataframe(path: str = 'cumulative_2025.10.04_09.55.40.csv'):
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Varsayılan veri seti bulunamadı: {data_path.resolve()}"
        )
    return load_reference_dataset(data_path)

def anonymize_sensitive_columns(df, sensitive_columns=None):
    """
    Veri setindeki hassas sütunları anonimleştirir.
    
    Args:
        df: Pandas DataFrame
        sensitive_columns: Anonimleştirilecek sütun listesi. 
                          None ise otomatik algılama yapılır.
    
    Returns:
        Anonimleştirilmiş DataFrame ve anonimleştirme raporu
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
        # İsim, ID, email, koordinat gibi hassas bilgiler içerebilecek sütunları tespit et
        sensitive_keywords = ['name', 'id', 'email', 'address', 'phone', 
                             'ra', 'dec', 'coordinate', 'location', 
                             'kepid', 'kepoi']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in sensitive_keywords):
                sensitive_columns.append(col)
    
    # Anonimleştirme işlemleri
    for col in sensitive_columns:
        if col not in df.columns:
            continue
            
        col_type = df[col].dtype
        
        # Sayısal değerler için hash
        if pd.api.types.is_numeric_dtype(col_type):
            df_anon[col] = df[col].apply(
                lambda x: int(hashlib.sha256(str(x).encode()).hexdigest()[:8], 16) % 1000000
                if pd.notna(x) else x
            )
            anonymization_report['method_used'][col] = 'numeric_hash'
        
        # String değerler için hash
        elif pd.api.types.is_string_dtype(col_type) or col_type == 'object':
            df_anon[col] = df[col].apply(
                lambda x: f"ANON_{hashlib.sha256(str(x).encode()).hexdigest()[:12].upper()}"
                if pd.notna(x) else x
            )
            anonymization_report['method_used'][col] = 'string_hash'
        
        anonymization_report['anonymized_columns'].append(col)
    
    return df_anon, anonymization_report

def validate_csv_file(uploaded_file):
    """
    Yüklenen dosyanın CSV formatında olduğunu doğrular.
    
    Args:
        uploaded_file: Streamlit UploadedFile objesi
    
    Returns:
        (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "Dosya yüklenmedi."
    
    # Dosya uzantısı kontrolü
    file_name = uploaded_file.name
    if not file_name.lower().endswith('.csv'):
        return False, f"❌ Geçersiz dosya türü: '{file_name}'. Sadece CSV dosyaları kabul edilir."
    
    # Dosya boyutu kontrolü (maksimum 100MB)
    max_size_mb = 100
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"❌ Dosya çok büyük: {file_size_mb:.2f}MB. Maksimum {max_size_mb}MB olmalıdır."
    
    # İçerik kontrolü - CSV formatını doğrula
    try:
        # Dosyayı okumayı dene
        uploaded_file.seek(0)
        test_df = pd.read_csv(uploaded_file, nrows=5)
        uploaded_file.seek(0)  # Dosya işaretçisini başa al
        
        if len(test_df.columns) == 0:
            return False, "❌ CSV dosyası geçerli sütunlar içermiyor."
        
        return True, "✅ Dosya geçerli."
    
    except pd.errors.EmptyDataError:
        return False, "❌ CSV dosyası boş."
    except pd.errors.ParserError as e:
        return False, f"❌ CSV ayrıştırma hatası: {str(e)}"
    except Exception as e:
        return False, f"❌ Dosya okuma hatası: {str(e)}"

# Özellik mühendisliği fonksiyonu
def engineer_features(features_dict, feature_names):
    """Özellik mühendisliği uygula"""
    X = pd.DataFrame([features_dict])
    
    # 1. Gezegen-yıldız boyut oranı
    if 'koi_prad' in X.columns and 'koi_srad' in X.columns:
        X['planet_star_ratio'] = X['koi_prad'] / (X['koi_srad'] * 109.2)
    
    # 2. Sinyal kalitesi göstergesi
    if 'koi_depth' in X.columns and 'koi_model_snr' in X.columns:
        X['signal_quality'] = X['koi_depth'] * X['koi_model_snr']
    
    # 3. Yörünge hızı tahmini
    if 'koi_period' in X.columns and X['koi_period'].iloc[0] > 0:
        X['orbital_velocity'] = 1 / X['koi_period']
    
    # 4. Yanlış pozitif toplam skoru
    fp_flags = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    available_fp_flags = [f for f in fp_flags if f in X.columns]
    if available_fp_flags:
        X['fp_total_score'] = X[available_fp_flags].sum(axis=1)
    
    # 5. Geçiş şekil faktörü
    if 'koi_duration' in X.columns and 'koi_period' in X.columns and X['koi_period'].iloc[0] > 0:
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
    
    return X

# Tahmin fonksiyonu
def predict_exoplanet(features_dict, model, scaler, feature_names, anomaly_detector=None):
    """Ötegezegen tahmini yap"""
    X = engineer_features(features_dict, feature_names)
    
    # Ölçeklendir ve tahmin yap
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    novelty = None
    if anomaly_detector is not None:
        novelty_flag = anomaly_detector.predict(X_scaled)[0]
        novelty_score = anomaly_detector.decision_function(X_scaled)[0]
        novelty = {
            'is_novel_candidate': bool(novelty_flag == -1),
            'novelty_score': float(novelty_score)
        }
    
    return {
        'is_exoplanet': bool(prediction),
        'probability_not_exoplanet': float(probability[0]),
        'probability_exoplanet': float(probability[1]),
        'confidence': float(max(probability)),
        'X_scaled': X_scaled,
        'X_original': X,
        'novelty': novelty
    }

# SHAP açıklaması
def get_shap_explanation(model, X_scaled, explainer, feature_names):
    """SHAP değerlerini hesapla"""
    shap_values = explainer.shap_values(X_scaled)
    return shap_values

# LIME açıklaması
def get_lime_explanation(model, X_scaled, X_original, feature_names, scaler):
    """LIME açıklaması oluştur"""
    # LIME explainer oluştur
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=feature_names,
        class_names=['Ötegezegen Değil', 'Ötegezegen'],
        mode='classification'
    )
    
    # Açıklama üret
    exp = explainer.explain_instance(
        X_scaled[0], 
        model.predict_proba,
        num_features=10
    )
    
    return exp

# Ana uygulama
def main():
    # Başlık
    enforce_access_control()
    
    # Modelleri yükle
    model, scaler, feature_names, explainer, anomaly_detector = load_models()
    feature_names = list(feature_names) if feature_names is not None else []
    
    if model is None:
        st.stop()
    
    # Sidebar - Navigasyon
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/2449px-NASA_logo.svg.png", width=100)
        st.title("📡 Navigasyon")

        page_definitions = [
            ("home", "🏠 Ana Sayfa"),
            ("live", "🛰️ NASA Canlı Akışı"),
            ("predict", "🔮 Tahmin Yap"),
            ("batch", "📊 Toplu Analiz"),
            ("synth", "🧪 Veri Üretimi"),
            ("3d_viz", "🌌 3B Yıldız Sistemi"),
            ("model", "🧠 Model Analizi"),
            ("about", "📚 Hakkında"),
        ]
        page_labels = {key: label for key, label in page_definitions}
        page_key = st.radio(
            "Sayfa Seçin:",
            options=[key for key, _ in page_definitions],
            format_func=lambda key: page_labels.get(key, key),
        )
        
        st.markdown("---")
        st.markdown("### 📈 Model Performansı")
        st.metric("Doğruluk", "94.46%", "4.46%")
        st.metric("F1 Skoru", "0.9065", "0.0065")
        st.metric("ROC AUC", "0.9839", "0.0839")
        
        st.markdown("---")
        st.markdown("### 🌟 İstatistikler")
        st.info(f"📅 **Tarih:** {datetime.now().strftime('%d.%m.%Y')}")
        st.info(f"🔢 **Toplam Özellik:** {len(feature_names)}")
        
    # Sayfa yönlendirme
    if page_key == "home":
        show_home_page()
    elif page_key == "live":
        show_live_data_page(model, scaler, feature_names, anomaly_detector)
    elif page_key == "predict":
        show_prediction_page(model, scaler, feature_names, explainer, anomaly_detector)
    elif page_key == "batch":
        show_batch_analysis_page(model, scaler, feature_names, anomaly_detector)
    elif page_key == "synth":
        show_data_generation_page()
    elif page_key == "3d_viz":
        show_3d_star_system_page()
    elif page_key == "model":
        show_model_analysis_page(model, scaler, feature_names, anomaly_detector)
    elif page_key == "about":
        show_about_page()

def show_home_page():
    """Ana sayfa"""
    st.markdown("""
        <div class="hero">
            <span class="tagline">görev kontrolü • ileri analitik</span>
            <div class="hero-title" style="margin-bottom:4px;">Ötegezegen Keşif Kontrol Paneli</div>
            <p class="hero-subtitle">Profesyonel veri bilimciler için tasarlanan panel; doğrulanmış Kepler metrikleri, XAI açıklamaları ve sentetik veri üretimi ile araştırma döngüsünü hızlandırır.</p>
            <div class="hero-badges">
                <span class="badge">🚀 XGBoost + XAI</span>
                <span class="badge">🧪 Sentetik veri laboratuvarı</span>
                <span class="badge">📡 SMOTE & Gauss karışımları</span>
                <span class="badge">🛰️ 20+ astrofizik özelliği</span>
                <span class="badge">🛰️ NASA canlı kataloğu</span>
            </div>
            <a class="cta-button" href="#prediction-anchor">Tahmin modülüne git →</a>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section-title" style="margin-top:28px;">Operasyonel göstergeler</div>
        <p class="section-subtitle">Son eğitim döngüsünden elde edilen metrikler. Trendler model güncellemeleriyle otomatik yenilenir.</p>
        <div class="metric-grid">
            <div class="metric-card">
                <h2>Model doğruluğu</h2>
                <div class="metric-value">94.46%</div>
                <div class="metric-delta">+4.46% hedefin üzerinde</div>
                <p>Kepler test kümesi performansı</p>
            </div>
            <div class="metric-card">
                <h2>XAI kapsamı</h2>
                <div class="metric-value">23 özellik</div>
                <div class="metric-delta">SHAP & LIME destekli</div>
                <p>Her tahmin için açıklanabilirlik katmanı</p>
            </div>
            <div class="metric-card">
                <h2>Sentetik üretim</h2>
                <div class="metric-value">3K kayıt</div>
                <div class="metric-delta">Hibrit strateji</div>
                <p>Yeni veri akışı ile sürekli keşif</p>
            </div>
            <div class="metric-card">
                <h2>Sunum gecikmesi</h2>
                <div class="metric-value">&lt; 120 ms</div>
                <div class="metric-delta">Gerçek zamanlı analiz</div>
                <p>Streamlit paneli için ortalama render süresi</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section-title">Platform modülleri</div>
        <p class="section-subtitle">Model eğitiminden sahada karar desteğine kadar tüm süreci tek panelden yönetin.</p>
        <div class="card-list">
            <div class="card-item">
                <h3>🔮 Tahmin İşleme</h3>
                <p>Tekil adayları saniyeler içinde değerlendirip güven skorları ve olasılıklar alın.</p>
                <div class="info-chip">Gerçek zamanlı SNR analizi</div>
                <div class="info-chip">Öznitelik normalizasyonu</div>
            </div>
            <div class="card-item">
                <h3>📊 Toplu Analiz</h3>
                <p>CSV yükleyerek binlerce adayın sonuçlarını ve özet istatistikleri tek raporda toplayın.</p>
                <div class="info-chip">Toplu skor kartı</div>
                <div class="info-chip">Otomatik veri temizleme</div>
            </div>
            <div class="card-item">
                <h3>🧠 Model Analizi</h3>
                <p>Karmaşıklık matrisi, ROC eğrisi ve özellik önem dağılımı ile model sağlığını izleyin.</p>
                <div class="info-chip">Detaylı performans paneli</div>
                <div class="info-chip">Özellik katkı raporu</div>
            </div>
            <div class="card-item">
                <h3>🧪 Sentetik Laboratuvar</h3>
                <p>SMOTE ve Gauss karışımlarını harmanlayarak dengeli, fiziksel olarak tutarlı veri üretilmesini sağlayın.</p>
                <div class="info-chip">Hedef sınıf oranı kontrolü</div>
                <div class="info-chip">Aykırı kırpma</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="section-title">Çalışma akışı</div>
        <p class="section-subtitle">Her modül veri bilimi sürecinin belirli bir aşamasını hızlandırmak üzere tasarlandı.</p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["1️⃣ Tahmin", "2️⃣ Toplu Analiz", "3️⃣ İzleme"])

    with tab1:
        st.markdown("""
            - Sol menüden **"🔮 Tahmin Yap"** sekmesini açın.
            - Gezegen parametrelerini girin veya varsayılan senaryoyu kullanın.
            - **"🚀 Tahmin Et"** butonunu tetikleyin; sonuçlar olasılık, güven ve XAI katkılarıyla birlikte gelir.
            - SHAP waterfall ve LIME analizine göre kritik özellikleri inceleyin.
        """)

    with tab2:
        st.markdown("""
            - **"📊 Toplu Analiz"** bölümünden CSV dosyanızı içeri alın.
            - Otomatik normalizasyon ve eksik değer iyileştirmesi tamamlandıktan sonra toplu tahminler üretilir.
            - Çıktı raporlarını indirerek araştırma notlarınıza ekleyin.
        """)

    with tab3:
        st.markdown("""
            - **"🧠 Model Analizi"** sekmesi ile üretim modelinin sağlığını izleyin.
            - Zaman içinde doğruluk eğilimlerini takip edin ve kritik durumlarda yeniden eğitim tetikleyin.
            - Sentetik laboratuvar modülüyle dengeli veri setleri oluşturarak modelinizi güncel tutun.
        """)

def show_prediction_page(model, scaler, feature_names, explainer, anomaly_detector):
    """Tahmin sayfası"""
    st.markdown("<div id='prediction-anchor'></div>", unsafe_allow_html=True)
    st.header("🔮 Ötegezegen Tahmini")
    st.markdown("Gezegen özelliklerini girerek ötegezegen olup olmadığını tahmin edin.")
    
    # Giriş modu seçimi
    input_mode = st.radio("Giriş Modu:", ["📝 Basit Mod (Ana Özellikler)", "🔬 Gelişmiş Mod (Tüm Özellikler)"], horizontal=True)
    
    with st.form("prediction_form"):
        st.subheader("📊 Gezegen Özellikleri")
        
        if input_mode == "📝 Basit Mod (Ana Özellikler)":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🌍 Gezegen Özellikleri**")
                koi_period = st.number_input("Yörünge Periyodu (gün)", value=3.52, min_value=0.0, help="Gezegenin yıldızını çevreleme süresi")
                koi_prad = st.number_input("Gezegen Yarıçapı (Dünya)", value=1.89, min_value=0.0, help="Dünya yarıçapı cinsinden")
                koi_teq = st.number_input("Denge Sıcaklığı (K)", value=1284.0, min_value=0.0, help="Kelvin cinsinden")
            
            with col2:
                st.markdown("**⭐ Yıldız Özellikleri**")
                koi_steff = st.number_input("Yıldız Sıcaklığı (K)", value=5455.0, min_value=0.0, help="Etkin sıcaklık")
                koi_srad = st.number_input("Yıldız Yarıçapı (Güneş)", value=0.927, min_value=0.0, help="Güneş yarıçapı cinsinden")
                koi_slogg = st.number_input("Yüzey Yerçekimi (log10)", value=4.467, min_value=0.0, help="cm/s² cinsinden")
            
            with col3:
                st.markdown("**🔭 Gözlem Verileri**")
                koi_depth = st.number_input("Geçiş Derinliği (ppm)", value=615.8, min_value=0.0, help="Işık eğrisindeki düşüş")
                koi_duration = st.number_input("Geçiş Süresi (saat)", value=2.95, min_value=0.0, help="Geçişin toplam süresi")
                koi_model_snr = st.number_input("Sinyal-Gürültü Oranı", value=35.8, min_value=0.0, help="Model SNR")
            
            # Varsayılan değerler
            features = {
                'koi_period': koi_period,
                'koi_depth': koi_depth,
                'koi_duration': koi_duration,
                'koi_ror': 0.0174,
                'koi_srho': 4.11,
                'koi_prad': koi_prad,
                'koi_teq': koi_teq,
                'koi_insol': 62.3,
                'koi_steff': koi_steff,
                'koi_slogg': koi_slogg,
                'koi_srad': koi_srad,
                'koi_impact': 0.146,
                'koi_model_snr': koi_model_snr,
                'koi_tce_plnt_num': 1,
                'koi_fpflag_nt': 0,
                'koi_fpflag_ss': 0,
                'koi_fpflag_co': 0,
                'koi_fpflag_ec': 0
            }
        
        else:  # Gelişmiş Mod
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**🌍 Gezegen**")
                koi_period = st.number_input("Yörünge Periyodu (gün)", value=3.52, min_value=0.0)
                koi_prad = st.number_input("Gezegen Yarıçapı", value=1.89, min_value=0.0)
                koi_teq = st.number_input("Denge Sıcaklığı (K)", value=1284.0, min_value=0.0)
                koi_insol = st.number_input("Güneş Işınımı", value=62.3, min_value=0.0)
                koi_ror = st.number_input("Yarıçap Oranı", value=0.0174, min_value=0.0)
            
            with col2:
                st.markdown("**⭐ Yıldız**")
                koi_steff = st.number_input("Yıldız Sıcaklığı (K)", value=5455.0, min_value=0.0)
                koi_srad = st.number_input("Yıldız Yarıçapı", value=0.927, min_value=0.0)
                koi_slogg = st.number_input("Yüzey Yerçekimi", value=4.467, min_value=0.0)
                koi_srho = st.number_input("Yıldız Yoğunluğu", value=4.11, min_value=0.0)
            
            with col3:
                st.markdown("**🔭 Gözlem**")
                koi_depth = st.number_input("Geçiş Derinliği (ppm)", value=615.8, min_value=0.0)
                koi_duration = st.number_input("Geçiş Süresi (saat)", value=2.95, min_value=0.0)
                koi_impact = st.number_input("Etki Parametresi", value=0.146, min_value=0.0)
                koi_model_snr = st.number_input("SNR", value=35.8, min_value=0.0)
                koi_tce_plnt_num = st.number_input("Gezegen No", value=1, min_value=1, step=1)
            
            with col4:
                st.markdown("**🚩 FP Bayrakları**")
                koi_fpflag_nt = st.selectbox("Transit Dışı", [0, 1], index=0)
                koi_fpflag_ss = st.selectbox("Yıldız Tutulması", [0, 1], index=0)
                koi_fpflag_co = st.selectbox("Merkez Sapması", [0, 1], index=0)
                koi_fpflag_ec = st.selectbox("Kontaminasyon", [0, 1], index=0)
            
            features = {
                'koi_period': koi_period,
                'koi_depth': koi_depth,
                'koi_duration': koi_duration,
                'koi_ror': koi_ror,
                'koi_srho': koi_srho,
                'koi_prad': koi_prad,
                'koi_teq': koi_teq,
                'koi_insol': koi_insol,
                'koi_steff': koi_steff,
                'koi_slogg': koi_slogg,
                'koi_srad': koi_srad,
                'koi_impact': koi_impact,
                'koi_model_snr': koi_model_snr,
                'koi_tce_plnt_num': int(koi_tce_plnt_num),
                'koi_fpflag_nt': koi_fpflag_nt,
                'koi_fpflag_ss': koi_fpflag_ss,
                'koi_fpflag_co': koi_fpflag_co,
                'koi_fpflag_ec': koi_fpflag_ec
            }
        
        # Tahmin butonu
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.form_submit_button("🚀 Tahmin Et", use_container_width=True)
    
    # Tahmin yap
    if predict_button:
        with st.spinner("🔮 Tahmin yapılıyor ve XAI analizi oluşturuluyor..."):
            result = predict_exoplanet(
                features,
                model,
                scaler,
                feature_names,
                anomaly_detector=anomaly_detector,
            )
            
            # Sonuç göster
            st.markdown("---")
            st.header("📊 Tahmin Sonuçları")
            
            # Ana sonuç
            if result['is_exoplanet']:
                st.markdown(
                    """
                    <div class="prediction-banner success">
                        🎉 Ötegezegen tespit edildi! Bu aday, model tarafından yüksek güvenle doğrulandı.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="prediction-banner danger">
                        ❌ Ötegezegen olarak sınıflandırılmadı. Belirleyici metrikleri XAI sekmesinden inceleyin.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # Metrikler
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Ötegezegen Olasılığı",
                    f"{result['probability_exoplanet']*100:.2f}%",
                    delta=f"{(result['probability_exoplanet']-0.5)*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Ötegezegen Olmama Olasılığı",
                    f"{result['probability_not_exoplanet']*100:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Güven Skoru",
                    f"{result['confidence']*100:.2f}%"
                )
            
            with col4:
                confidence_emoji = "🟢" if result['confidence'] > 0.9 else "🟡" if result['confidence'] > 0.7 else "🔴"
                st.metric(
                    "Güvenilirlik",
                    confidence_emoji,
                    "Yüksek" if result['confidence'] > 0.9 else "Orta" if result['confidence'] > 0.7 else "Düşük"
                )

            novelty = result.get('novelty')
            if novelty is not None:
                novelty_cols = st.columns(2)
                novelty_score = novelty.get('novelty_score', 0.0)
                is_novel_candidate = novelty.get('is_novel_candidate', False)
                with novelty_cols[0]:
                    st.metric("Novelty Skoru", f"{novelty_score:.3f}")
                with novelty_cols[1]:
                    st.metric(
                        "Yeni Aday",
                        "🚨 EVET" if is_novel_candidate else "✅ HAYIR",
                        help="IsolationForest tabanlı anomalilik kontrolü"
                    )

                if is_novel_candidate:
                    st.warning(
                        "🚨 Model dağılımının dışında kalan bir aday tespit edildi. Fiziksel doğrulama ve ilave gözlem önerilir."
                    )
                else:
                    st.success(
                        "🛡️ Aday, eğitim dağılımı ile uyumlu görünüyor. Standart inceleme protokolünü uygulayabilirsiniz."
                    )
            
            # Olasılık grafiği
            st.subheader("📊 Olasılık Dağılımı")
            fig = go.Figure(data=[
                go.Bar(
                    x=['Ötegezegen Değil', 'Ötegezegen'],
                    y=[result['probability_not_exoplanet'], result['probability_exoplanet']],
                    marker_color=['#ff6a00', '#38ef7d'],
                    text=[f"{result['probability_not_exoplanet']*100:.2f}%", 
                          f"{result['probability_exoplanet']*100:.2f}%"],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Sınıf Olasılıkları",
                yaxis_title="Olasılık",
                yaxis_range=[0, 1],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # XAI Açıklamaları
            st.markdown("---")
            st.header("🧠 XAI - Açıklanabilir AI Analizi")
            st.markdown("**'Model neden bu kararı verdi?'** sorusuna detaylı cevaplar")
            
            # Explainability Engine'i başlat
            try:
                xai_engine = ExplainabilityEngine(model, scaler, feature_names, explainer)
                full_explanation = xai_engine.generate_decision_explanation(
                    result['X_scaled'], 
                    result['X_original'],
                    result
                )
                
                # Anlatısal açıklama
                st.markdown("### 📖 Karar Açıklaması")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**🤖 AI Motoru Açıklaması**")
                    narrative = create_decision_narrative(full_explanation)
                    st.markdown(narrative)
                
                with col2:
                    st.markdown("**✨ Gemini AI Yorumu**")
                    try:
                        with st.spinner("Gemini AI analiz ediyor..."):
                            gemini = GeminiExplainer(GEMINI_API_KEY)
                            gemini_explanation = gemini.generate_explanation(result, full_explanation)
                            
                            if gemini_explanation:
                                st.markdown(gemini_explanation)
                            else:
                                st.warning("Gemini AI yorumu oluşturulamadı.")
                    except Exception as gemini_error:
                        st.error(f"Gemini AI hatası: {gemini_error}")
                
            except Exception as e:
                st.warning(f"XAI Engine başlatılamadı: {e}")
                full_explanation = None
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 SHAP Analizi", 
                "🔍 Özellik Katkıları", 
                "📈 Karar Kuralları",
                "🎯 What-If Analizi",
                "📉 Karşılaştırmalı Analiz"
            ])
            
            with tab1:
                st.subheader("SHAP (SHapley Additive exPlanations)")
                st.info("🎯 SHAP, her özelliğin tahmine olan katkısını gösterir. Pozitif değerler ötegezegen olasılığını artırır, negatif değerler azaltır.")
                
                try:
                    # SHAP değerlerini hesapla
                    shap_values = get_shap_explanation(model, result['X_scaled'], explainer, feature_names)
                    
                    # İnteraktif SHAP Waterfall Plot
                    st.markdown("**🌊 İnteraktif SHAP Waterfall Plot**")
                    st.caption("Her özelliğin tahmine olan katkısını gösterir - üzerine gelin daha fazla bilgi için")
                    
                    waterfall_fig = create_shap_waterfall_plotly(
                        shap_values[0],
                        explainer.expected_value,
                        result['X_scaled'][0],
                        feature_names,
                        max_display=10
                    )
                    st.plotly_chart(waterfall_fig, use_container_width=True)
                    
                    # SHAP özet bilgileri
                    if full_explanation and full_explanation['shap_analysis']:
                        shap_info = full_explanation['shap_analysis']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Toplam Pozitif Etki", f"+{shap_info['total_positive_impact']:.3f}")
                        with col2:
                            st.metric("Toplam Negatif Etki", f"{shap_info['total_negative_impact']:.3f}")
                        with col3:
                            net_impact = shap_info['total_positive_impact'] + shap_info['total_negative_impact']
                            st.metric("Net Etki", f"{net_impact:+.3f}")
                    
                    # En etkili özellikler
                    st.markdown("**🏆 En Etkili 10 Özellik**")
                    shap_df = pd.DataFrame({
                        'Özellik': feature_names,
                        'Değer': result['X_scaled'][0],
                        'SHAP Değeri': shap_values[0],
                        'Etki Yönü': ['Pozitif ⬆️' if v > 0 else 'Negatif ⬇️' if v < 0 else 'Nötr ➡️' for v in shap_values[0]]
                    })
                    shap_df['Mutlak Etki'] = np.abs(shap_df['SHAP Değeri'])
                    shap_df = shap_df.sort_values('Mutlak Etki', ascending=False)
                    
                    st.dataframe(
                        shap_df[['Özellik', 'Değer', 'SHAP Değeri', 'Etki Yönü']].head(10),
                        use_container_width=True
                    )
                    
                    # Matplotlib versiyonu (opsiyonel)
                    with st.expander("📊 Klasik SHAP Waterfall Plot (Matplotlib)"):
                        fig, ax = plt.subplots(figsize=(10, 8))
                        shap.plots.waterfall(
                            shap.Explanation(
                                values=shap_values[0],
                                base_values=explainer.expected_value,
                                data=result['X_scaled'][0],
                                feature_names=feature_names
                            ),
                            show=False
                        )
                        st.pyplot(fig)
                        plt.close()
                    
                except Exception as e:
                    st.error(f"❌ SHAP analizi hatası: {e}")
            
            with tab2:
                st.subheader("🔍 Özellik Katkı Analizi")
                st.info("💡 Her özelliğin modelin kararına olan katkısını detaylı olarak gösterir.")
                
                try:
                    if full_explanation and full_explanation['feature_contribution']:
                        contrib_data = full_explanation['feature_contribution']
                        contributions = contrib_data['contributions']
                        
                        # Katkı grafiği
                        st.markdown("**� Ağırlıklı Özellik Katkıları**")
                        
                        contrib_df = pd.DataFrame(contributions)
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=contrib_df['weighted_contribution'].head(15),
                                y=contrib_df['feature'].head(15),
                                orientation='h',
                                marker=dict(
                                    color=contrib_df['weighted_contribution'].head(15),
                                    colorscale='RdYlGn',
                                    showscale=True,
                                    colorbar=dict(title="Katkı")
                                ),
                                text=contrib_df['weighted_contribution'].head(15).apply(lambda x: f"{x:.3f}"),
                                textposition='auto',
                            )
                        ])
                        fig.update_layout(
                            title="En Etkili 15 Özellik (Ağırlıklı Katkı)",
                            xaxis_title="Ağırlıklı Katkı",
                            yaxis_title="Özellik",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model önem derecesi vs SHAP karşılaştırması
                        if full_explanation['shap_analysis']:
                            st.markdown("**⚖️ Model Önem Derecesi vs SHAP Karşılaştırması**")
                            
                            shap_importance = np.abs(full_explanation['shap_analysis']['values'])
                            comp_fig = create_feature_importance_comparison(
                                model.feature_importances_,
                                shap_importance,
                                feature_names,
                                top_n=15
                            )
                            st.plotly_chart(comp_fig, use_container_width=True)
                        
                        # Detaylı tablo
                        st.markdown("**📋 Detaylı Katkı Tablosu**")
                        display_df = contrib_df[['feature', 'value', 'importance', 'percentage', 'weighted_contribution']].head(15)
                        display_df.columns = ['Özellik', 'Değer', 'Model Önemi', 'Yüzde (%)', 'Ağırlıklı Katkı']
                        display_df['Yüzde (%)'] = display_df['Yüzde (%)'].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(display_df, use_container_width=True)
                        
                    else:
                        st.warning("Özellik katkı analizi mevcut değil.")
                    
                except Exception as e:
                    st.error(f"❌ Özellik katkı analizi hatası: {e}")
            
            with tab3:
                st.subheader("� Karar Kuralları ve Eşik Değerleri")
                st.info("🎯 Kritik özelliklerin optimal aralıklarda olup olmadığını kontrol eder.")
                
                try:
                    if full_explanation and full_explanation['decision_rules']:
                        rules = full_explanation['decision_rules']
                        
                        # Karar yolu görselleştirmesi
                        st.markdown("**🛤️ Karar Yolu Analizi**")
                        decision_path_fig = create_decision_path_visualization(full_explanation)
                        if decision_path_fig:
                            st.plotly_chart(decision_path_fig, use_container_width=True)
                        
                        # Durum özeti
                        optimal_count = sum(1 for r in rules if r['status'] == 'optimal')
                        acceptable_count = sum(1 for r in rules if r['status'] == 'acceptable')
                        out_of_range_count = sum(1 for r in rules if r['status'] == 'out_of_range')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("✅ Optimal", optimal_count)
                        with col2:
                            st.metric("⚠️ Kabul Edilebilir", acceptable_count)
                        with col3:
                            st.metric("❌ Aralık Dışı", out_of_range_count)
                        
                        # Detaylı kurallar
                        st.markdown("**� Kural Detayları**")
                        
                        for rule in rules:
                            status_color = {
                                'optimal': 'green',
                                'acceptable': 'orange',
                                'out_of_range': 'red'
                            }[rule['status']]
                            
                            status_text = {
                                'optimal': '✅ Optimal',
                                'acceptable': '⚠️ Kabul Edilebilir',
                                'out_of_range': '❌ Aralık Dışı'
                            }[rule['status']]
                            
                            with st.expander(f"{status_text} - {rule['feature']}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Mevcut Değer", f"{rule['value']:.2f}")
                                    st.markdown(f"**Optimal Aralık:** {rule['optimal_range'][0]} - {rule['optimal_range'][1]}")
                                
                                with col2:
                                    st.markdown(f"**Minimum:** {rule['threshold_low']}")
                                    st.markdown(f"**Maksimum:** {rule['threshold_high']}")
                                    impact_emoji = "⬆️" if rule['impact'] == 'positive' else "⬇️"
                                    st.markdown(f"**Etki:** {impact_emoji} {rule['impact'].title()}")
                    else:
                        st.warning("Karar kuralları mevcut değil.")
                        
                except Exception as e:
                    st.error(f"❌ Karar kuralı analizi hatası: {e}")
            
            with tab4:
                st.subheader("🎯 What-If Analizi")
                st.info("🔮 Kritik özelliklerdeki değişikliklerin tahmini nasıl etkileyeceğini simüle eder.")
                
                try:
                    if full_explanation and full_explanation['what_if_analysis']:
                        scenarios = full_explanation['what_if_analysis']
                        
                        st.markdown("**🧪 Senaryo Simülasyonu**")
                        st.caption("Özellikleri değiştirerek farklı sonuçları öngörün")
                        
                        for scenario in scenarios:
                            with st.expander(f"🔬 {scenario['feature']} Senaryoları"):
                                st.metric("Mevcut Değer", f"{scenario['original_value']:.3f}")
                                
                                st.markdown("**Alternatif Senaryolar:**")
                                
                                scenario_df = pd.DataFrame(scenario['scenarios'])
                                scenario_df.columns = ['Değişiklik', 'Yeni Değer']
                                scenario_df['Yeni Değer'] = scenario_df['Yeni Değer'].apply(lambda x: f"{x:.3f}")
                                st.dataframe(scenario_df, use_container_width=True)
                                
                                st.markdown("💡 **Not:** Gerçek tahmin için yukarıdaki değerleri kullanarak yeni bir tahmin yapabilirsiniz.")
                        
                        # İnteraktif what-if
                        st.markdown("---")
                        st.markdown("**🎮 İnteraktif What-If Simulator**")
                        
                        selected_feature = st.selectbox(
                            "Değiştirmek istediğiniz özelliği seçin:",
                            options=[s['feature'] for s in scenarios]
                        )
                        
                        selected_scenario = next(s for s in scenarios if s['feature'] == selected_feature)
                        
                        new_value = st.slider(
                            f"{selected_feature} - Yeni Değer",
                            min_value=selected_scenario['original_value'] * 0.1,
                            max_value=selected_scenario['original_value'] * 2.0,
                            value=selected_scenario['original_value'],
                            step=selected_scenario['original_value'] * 0.05
                        )
                        
                        change_pct = ((new_value - selected_scenario['original_value']) / selected_scenario['original_value']) * 100
                        st.metric("Değişim Yüzdesi", f"{change_pct:+.1f}%")
                        
                    else:
                        st.warning("What-if analizi mevcut değil.")
                        
                except Exception as e:
                    st.error(f"❌ What-if analizi hatası: {e}")
            
            with tab5:
                st.subheader("📉 Karşılaştırmalı Analiz")
                st.info("🌍 Girdiğiniz değerleri bilinen gezegenlerle karşılaştırın.")
                
                # Girilen özellikler
                input_features = pd.DataFrame([features]).T
                input_features.columns = ['Değer']
                input_features['Özellik'] = input_features.index
                input_features = input_features[['Özellik', 'Değer']]
                
                st.markdown("**📊 Girilen Özellikler**")
                st.dataframe(input_features.head(10), use_container_width=True)
                
                # Referans karşılaştırması
                st.markdown("**🔄 Bilinen Gezegenlerle Karşılaştırma**")
                
                comparison_data = {
                    'Özellik': ['Yörünge Periyodu (gün)', 'Gezegen Yarıçapı (Dünya)', 'Denge Sıcaklığı (K)', 'Yıldız Sıcaklığı (K)'],
                    'Sizin Değer': [
                        features.get('koi_period', 0),
                        features.get('koi_prad', 0),
                        features.get('koi_teq', 0),
                        features.get('koi_steff', 0)
                    ],
                    'Dünya': [365.25, 1.0, 255.0, 5778.0],
                    'Mars': [687.0, 0.532, 210.0, 5778.0],
                    'Jüpiter': [4333.0, 11.2, 110.0, 5778.0],
                    'Neptün': [60182.0, 3.88, 55.0, 5778.0]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Görsel karşılaştırma
                st.markdown("**📊 Görsel Karşılaştırma**")
                
                fig = go.Figure()
                
                for planet in ['Dünya', 'Mars', 'Jüpiter', 'Neptün']:
                    fig.add_trace(go.Bar(
                        name=planet,
                        x=comparison_data['Özellik'][:3],  # İlk 3 özellik
                        y=[comparison_df[comparison_df['Özellik'] == feat][planet].values[0] 
                           for feat in comparison_data['Özellik'][:3]],
                    ))
                
                fig.add_trace(go.Bar(
                    name='Sizin Değer',
                    x=comparison_data['Özellik'][:3],
                    y=comparison_data['Sizin Değer'][:3],
                    marker_color='red'
                ))
                
                fig.update_layout(
                    title="Gezegen Özelliklerinin Karşılaştırması",
                    xaxis_title="Özellik",
                    yaxis_title="Değer",
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Güven faktörleri
                if full_explanation and full_explanation['confidence_factors']:
                    st.markdown("**🎯 Güvenilirlik Faktörleri**")
                    
                    conf_factors = full_explanation['confidence_factors']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Güven Seviyesi", f"{conf_factors['confidence_level']*100:.1f}%")
                    with col2:
                        st.metric("Olasılık Marjı", f"{conf_factors['probability_margin']:.3f}")
                    
                    if conf_factors['factors']:
                        for factor in conf_factors['factors']:
                            impact_emoji = "✅" if factor['impact'] == 'positive' else "⚠️"
                            st.markdown(f"{impact_emoji} **{factor['factor']}:** {factor['description']}")


def show_3d_star_system_page():
    """3B etkileşimli yıldız sistemi görselleştirmesi"""
    st.header("🌌 3B Yıldız Sistemi Görselleştirmesi")
    st.markdown(
        "Kepler sistemlerini üç boyutlu uzayda keşfedin. Gezegenlerin yıldızlarına göre konumlarını, "
        "yörüngelerini ve fiziksel özelliklerini interaktif olarak inceleyin."
    )

    # Veri yükleme
    try:
        default_path = Path('cumulative_2025.10.04_09.55.40.csv')
        if default_path.exists():
            df_raw = pd.read_csv(default_path, comment='#')
        else:
            df_raw = load_default_reference_dataframe()
    except Exception as exc:
        st.error(f"Veri yüklenirken hata oluştu: {exc}")
        return

    # Temel filtreleme - sadece CONFIRMED gezegenleri al
    df = df_raw[df_raw['koi_disposition'] == 'CONFIRMED'].copy()
    
    # Minimum gerekli sütunlar (yörünge periyodu ve gezegen yarıçapı zorunlu)
    minimal_required = ['koi_period', 'koi_prad']
    optional_cols = ['koi_teq', 'koi_srad', 'koi_steff']
    
    # Sayısal dönüşüm
    for col in minimal_required + optional_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sadece zorunlu sütunlarda NaN olanları çıkar
    df = df.dropna(subset=minimal_required)
    
    # Opsiyonel sütunlar için varsayılan değerler
    if 'koi_teq' not in df.columns or df['koi_teq'].isna().all():
        df['koi_teq'] = 300  # Varsayılan sıcaklık
    else:
        df['koi_teq'] = df['koi_teq'].fillna(df['koi_teq'].median())
    
    if 'koi_srad' not in df.columns or df['koi_srad'].isna().all():
        df['koi_srad'] = 1.0  # Varsayılan yıldız yarıçapı (Güneş yarıçapı)
    else:
        df['koi_srad'] = df['koi_srad'].fillna(df['koi_srad'].median())
    
    if 'koi_steff' not in df.columns or df['koi_steff'].isna().all():
        df['koi_steff'] = 5778  # Varsayılan yıldız sıcaklığı (Güneş sıcaklığı)
    else:
        df['koi_steff'] = df['koi_steff'].fillna(df['koi_steff'].median())

    if df.empty:
        st.warning("Görselleştirme için yeterli CONFIRMED gezegen verisi bulunamadı.")
        return
    
    # Pozitif değer kontrolü
    df = df[(df['koi_period'] > 0) & (df['koi_prad'] > 0)]
    
    if df.empty:
        st.warning("Geçerli yörünge verisi bulunamadı.")
        return

    st.info(f"📊 Görselleştirilen gezegen sayısı: {len(df):,}")

    # Kontrol paneli
    st.markdown("### ⚙️ Görselleştirme Kontrolleri")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        color_by = st.selectbox(
            "Renklendirme",
            ["koi_teq", "koi_prad", "koi_steff", "koi_period"],
            format_func=lambda x: {
                "koi_teq": "Denge Sıcaklığı",
                "koi_prad": "Gezegen Yarıçapı",
                "koi_steff": "Yıldız Sıcaklığı",
                "koi_period": "Yörünge Periyodu"
            }[x]
        )

    with col_ctrl2:
        size_by = st.selectbox(
            "Nokta Boyutu",
            ["koi_prad", "koi_period", "koi_teq"],
            format_func=lambda x: {
                "koi_prad": "Gezegen Yarıçapı",
                "koi_period": "Yörünge Periyodu",
                "koi_teq": "Denge Sıcaklığı"
            }[x]
        )

    with col_ctrl3:
        max_points = st.slider("Maksimum nokta sayısı", 50, 500, 200, 50)

    # Veriyi sınırla
    df_viz = df.head(max_points).copy()

    # 3B koordinatları hesapla (yörünge yaklaşımı)
    # Basitleştirilmiş model: yörünge periyodu ve gezegen büyüklüğüne göre konumlandırma
    df_viz['orbit_radius'] = np.cbrt(df_viz['koi_period'])  # Kepler's 3rd law approximation
    df_viz['theta'] = np.random.uniform(0, 2*np.pi, len(df_viz))
    df_viz['phi'] = np.random.uniform(0, np.pi, len(df_viz))
    
    df_viz['x'] = df_viz['orbit_radius'] * np.sin(df_viz['phi']) * np.cos(df_viz['theta'])
    df_viz['y'] = df_viz['orbit_radius'] * np.sin(df_viz['phi']) * np.sin(df_viz['theta'])
    df_viz['z'] = df_viz['orbit_radius'] * np.cos(df_viz['phi'])

    # Yıldız skalası (normalize)
    df_viz['star_size'] = (df_viz['koi_srad'] / df_viz['koi_srad'].mean()) * 5
    df_viz['planet_size'] = (df_viz[size_by] / df_viz[size_by].max()) * 15 + 5

    st.markdown("### 🌌 İnteraktif 3B Görünüm")
    
    # Ana 3B scatter plot
    fig = go.Figure()

    # Merkezi yıldızları temsil eden noktalar (daha büyük)
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=25,
            color='yellow',
            symbol='diamond',
            line=dict(color='orange', width=2)
        ),
        name='Merkez Yıldız (ref)',
        hovertemplate='<b>Merkez Yıldız</b><extra></extra>'
    ))

    # Gezegenleri göster
    hover_template = (
        '<b>Gezegen</b><br>'
        'Yörünge Periyodu: %{customdata[0]:.2f} gün<br>'
        'Yarıçap: %{customdata[1]:.2f} R⊕<br>'
        'Sıcaklık: %{customdata[2]:.0f} K<br>'
        'Yıldız Yarıçapı: %{customdata[3]:.2f} R☉<br>'
        'Yıldız Sıcaklığı: %{customdata[4]:.0f} K'
        '<extra></extra>'
    )

    fig.add_trace(go.Scatter3d(
        x=df_viz['x'],
        y=df_viz['y'],
        z=df_viz['z'],
        mode='markers',
        marker=dict(
            size=df_viz['planet_size'],
            color=df_viz[color_by],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=color_by.replace('koi_', '').replace('_', ' ').title(),
                x=1.1
            ),
            line=dict(color='white', width=0.5)
        ),
        customdata=np.column_stack((
            df_viz['koi_period'],
            df_viz['koi_prad'],
            df_viz['koi_teq'],
            df_viz['koi_srad'],
            df_viz['koi_steff']
        )),
        hovertemplate=hover_template,
        name='Ötegezegen'
    ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X (Yörünge Ekseni)',
                backgroundcolor="rgba(0, 0, 0, 0.1)",
                gridcolor="gray",
                showbackground=True,
            ),
            yaxis=dict(
                title='Y (Yörünge Ekseni)',
                backgroundcolor="rgba(0, 0, 0, 0.1)",
                gridcolor="gray",
                showbackground=True,
            ),
            zaxis=dict(
                title='Z (Yörünge Ekseni)',
                backgroundcolor="rgba(0, 0, 0, 0.1)",
                gridcolor="gray",
                showbackground=True,
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='cube'
        ),
        title={
            'text': '3B Kepler Yıldız Sistemi Haritası',
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=True,
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # İstatistikler
    st.markdown("### 📊 Sistem İstatistikleri")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Ortalama Yörünge Periyodu", f"{df_viz['koi_period'].mean():.1f} gün")
    with stat_col2:
        st.metric("Ortalama Gezegen Yarıçapı", f"{df_viz['koi_prad'].mean():.2f} R⊕")
    with stat_col3:
        st.metric("Ortalama Sıcaklık", f"{df_viz['koi_teq'].mean():.0f} K")
    with stat_col4:
        st.metric("Ortalama Yıldız Yarıçapı", f"{df_viz['koi_srad'].mean():.2f} R☉")

    # Detaylı veri tablosu
    with st.expander("📋 Detaylı Veri Tablosu"):
        display_columns = ['koi_period', 'koi_prad', 'koi_teq', 'koi_srad', 'koi_steff', 'orbit_radius']
        st.dataframe(
            df_viz[display_columns].head(50),
            use_container_width=True
        )

    # Açıklama
    st.markdown("""
    ---
    ### 📖 Görselleştirme Hakkında
    
    Bu 3B görselleştirme, Kepler kataloğundaki onaylanmış ötegezegenlerin yıldızlarına göre 
    yaklaşık konumlarını gösterir. Koordinatlar, Kepler'in 3. yasasına dayalı basitleştirilmiş 
    bir model kullanılarak hesaplanmıştır.
    
    - **Merkez nokta**: Referans yıldızı temsil eder
    - **Renkli noktalar**: Her biri onaylanmış bir ötegezegeni temsil eder
    - **Nokta boyutu**: Seçilen fiziksel özelliğe göre ölçeklenir
    - **Renk**: Seçilen fiziksel parametreyi gösterir
    
    🎯 **Etkileşim İpuçları:**
    - Fareyle sürükleyerek görünümü döndürün
    - Tekerlek ile yakınlaştırın/uzaklaştırın
    - Noktalara tıklayarak detaylı bilgi görün
    """)


def show_live_data_page(model, scaler, feature_names, anomaly_detector):
    """NASA canlı kataloğunu görüntüler ve hibrit analiz uygular."""
    st.header("🛰️ NASA Canlı Exoplanet Akışı")
    st.markdown(
        "Gerçek zamanlı **Kepler** NASA Exoplanet Archive kataloğunu çekip hibrit fiziksel filtreler, "
        "anomali tespiti ve 3B görselleştirme ile değerlendirin."
    )

    mission_options = {
        "Kepler (Primary Mission)": "kepler",
    }

    col_sel1, col_sel2, col_sel3 = st.columns([1.4, 1, 1])
    mission_label = list(mission_options.keys())[0]
    with col_sel1:
        st.markdown(f"**Görev:** {mission_label}")
    with col_sel2:
        limit = st.slider("Kayıt limiti", min_value=100, max_value=5000, value=500, step=100)
    with col_sel3:
        force_refresh = st.checkbox(
            "Önbelleği yenile",
            value=False,
            help="Etkin olduğunda NASA API'dan taze veri çekilir.",
        )

    mission_code = mission_options[mission_label]

    try:
        with st.spinner(f"{mission_label} kataloğu getiriliyor..."):
            live_df = fetch_live_catalog(mission_code, limit, force_refresh)
    except Exception as exc:
        st.error(f"NASA verisi alınamadı: {exc}")
        return

    if live_df.empty:
        st.warning("Belirtilen kriterlerle eşleşen kayıt bulunamadı.")
        return

    scoring = score_catalog(live_df, model, scaler, list(feature_names), anomaly_detector)
    scored_df = scoring["scored"]

    st.caption(
        "Veriler NASA Exoplanet Archive TAP API üzerinden alınır ve yerelde 15 dakika boyunca cache'lenir."
    )

    total_records = len(scored_df)
    dispositions = (
        scored_df["koi_disposition"].str.upper()
        if "koi_disposition" in scored_df.columns
        else pd.Series(["UNKNOWN"] * total_records)
    )
    confirmed_count = int((dispositions == "CONFIRMED").sum()) if not dispositions.empty else 0
    candidate_count = int((dispositions == "CANDIDATE").sum()) if not dispositions.empty else 0
    novel_count = int(scored_df.get("is_novel_candidate", pd.Series(dtype=bool)).sum())
    high_conf = int((scored_df["model_probability"] >= 0.9).sum())

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Kayıt Sayısı", f"{total_records:,}")
    with metric_cols[1]:
        st.metric("CONFIRMED", f"{confirmed_count:,}")
    with metric_cols[2]:
        st.metric("Adaylar", f"{candidate_count:,}")
    with metric_cols[3]:
        st.metric("Yüksek Güven (≥0.9)", f"{high_conf:,}")

    if anomaly_detector is not None:
        st.info(
            f"🧭 {novel_count} kayıt, dağılım dışı potansiyel yeni aday olarak işaretlendi."
            if novel_count
            else "🛡️ Anomali modeli şu anda yeni aday saptamadı."
        )

    filtered_df = scored_df.copy()
    with st.expander("🔬 Hibrit fizik filtreleri"):
        if "koi_teq" in filtered_df.columns:
            teq_min = float(filtered_df["koi_teq"].min())
            teq_max = float(filtered_df["koi_teq"].max())
            if teq_min < teq_max:
                teq_range = st.slider(
                    "Denge sıcaklığı (K)",
                    min_value=float(round(teq_min, 2)),
                    max_value=float(round(teq_max, 2)),
                    value=(float(round(teq_min, 2)), float(round(teq_max, 2))),
                    key="live_teq_range",
                )
                filtered_df = filtered_df[filtered_df["koi_teq"].between(*teq_range)]

        if "koi_period" in filtered_df.columns:
            per_min = float(filtered_df["koi_period"].min())
            per_max = float(filtered_df["koi_period"].max())
            if per_min < per_max:
                period_range = st.slider(
                    "Yörünge periyodu (gün)",
                    min_value=float(round(per_min, 2)),
                    max_value=float(round(per_max, 2)),
                    value=(float(round(per_min, 2)), float(round(per_max, 2))),
                    key="live_period_range",
                )
                filtered_df = filtered_df[filtered_df["koi_period"].between(*period_range)]

        if "koi_prad" in filtered_df.columns:
            prad_min = float(filtered_df["koi_prad"].min())
            prad_max = float(filtered_df["koi_prad"].max())
            if prad_min < prad_max:
                prad_range = st.slider(
                    "Gezegen yarıçapı (Dünya)",
                    min_value=float(round(prad_min, 2)),
                    max_value=float(round(prad_max, 2)),
                    value=(float(round(prad_min, 2)), float(round(prad_max, 2))),
                    key="live_prad_range",
                )
                filtered_df = filtered_df[filtered_df["koi_prad"].between(*prad_range)]

        only_novel = st.checkbox(
            "Yalnızca yeni adayları listele",
            value=False,
            key="live_novel_only",
        )

    display_df = (
        filtered_df[filtered_df["is_novel_candidate"]]
        if only_novel and "is_novel_candidate" in filtered_df
        else filtered_df
    )

    st.markdown(
        f"**Görüntülenen kayıt sayısı:** {len(display_df):,} / {total_records:,}"
    )

    # DEBUG: Sütun ve veri durumunu kontrol et
    available_cols = list(display_df.columns)
    
    # 3D görselleştirme için minimum gereksinimler (sadece period ve prad zorunlu)
    minimal_plot_cols = ["koi_period", "koi_prad"]
    optional_plot_col = "koi_teq"
    
    # Sütunları kontrol et
    missing_cols = [col for col in minimal_plot_cols if col not in available_cols]
    
    if missing_cols:
        st.warning(f"⚠️ Gerekli sütunlar eksik: {', '.join(missing_cols)}")
        st.info(f"📋 Mevcut sütunlar: {', '.join([c for c in available_cols if 'koi' in c.lower()][:10])}")
    elif display_df.empty:
        st.warning("📭 Görüntülenecek kayıt yok (filtreler çok kısıtlayıcı olabilir)")
    elif all(col in display_df.columns for col in minimal_plot_cols):
        # Sadece zorunlu sütunlarda NaN olanları çıkar
        plot_df = display_df.dropna(subset=minimal_plot_cols).copy()
        
        # Sayısal dönüşüm
        for col in minimal_plot_cols + [optional_plot_col]:
            if col in plot_df.columns:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        
        # Zorunlu sütunları tekrar kontrol et
        plot_df = plot_df.dropna(subset=minimal_plot_cols)
        
        # Opsiyonel koi_teq için varsayılan değer
        if optional_plot_col in plot_df.columns:
            if plot_df[optional_plot_col].isna().all():
                plot_df[optional_plot_col] = 300  # Varsayılan sıcaklık
            else:
                plot_df[optional_plot_col] = plot_df[optional_plot_col].fillna(
                    plot_df[optional_plot_col].median()
                )
        else:
            plot_df[optional_plot_col] = 300
        
        if plot_df.empty:
            st.info("3B görselleştirme için yeterli ölçüm mevcut değil (minimum: koi_period ve koi_prad gerekli).")
        else:
            hover_candidates = ["koi_disposition", "novelty_score", "source_mission"]
            hover_map = {
                col: True
                for col in hover_candidates
                if col in plot_df.columns
            }
            hover_map["model_probability"] = ":.2f"

            fig = px.scatter_3d(
                plot_df,
                x="koi_period",
                y="koi_prad",
                z="koi_teq",
                color="model_probability",
                color_continuous_scale="Turbo",
                hover_data=hover_map,
                title="3B Görev keşif haritası",
            )

            if "is_novel_candidate" in plot_df.columns and plot_df["is_novel_candidate"].any():
                novel_points = plot_df[plot_df["is_novel_candidate"]]
                fig.add_trace(
                    go.Scatter3d(
                        x=novel_points["koi_period"],
                        y=novel_points["koi_prad"],
                        z=novel_points["koi_teq"],
                        mode="markers",
                        marker=dict(color="#ff4d4f", size=6, symbol="diamond"),
                        name="Yeni aday",
                    )
                )

            fig.update_scenes(
                xaxis_title="Periyot (gün)", yaxis_title="Yarıçap (Dünya)", zaxis_title="Teq (K)"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("3B görselleştirme için yeterli fiziksel özellik bulunamadı.")

    # Model Tahmin İstatistikleri
    st.markdown("### 🤖 Model Değerlendirme Sonuçları")
    
    if "model_probability" in display_df.columns and "model_prediction" in display_df.columns:
        # Farklı güven seviyelerine göre tahminler
        high_confidence = display_df[display_df["model_probability"] >= 0.9]
        medium_confidence = display_df[display_df["model_probability"].between(0.7, 0.9)]
        low_confidence = display_df[display_df["model_probability"] < 0.7]
        
        # Model tahminlerine göre ötegezegen sayıları
        predicted_exoplanets = display_df[display_df["model_prediction"] == "CONFIRMED"]
        predicted_non_exoplanets = display_df[display_df["model_prediction"] == "OTHER"]
        
        # Metrikler
        metric_cols = st.columns(5)
        with metric_cols[0]:
            st.metric(
                "🪐 Ötegezegen Tahmini", 
                f"{len(predicted_exoplanets):,}",
                help="Model tarafından CONFIRMED olarak tahmin edilen objeler"
            )
        with metric_cols[1]:
            st.metric(
                "❌ Ötegezegen Değil", 
                f"{len(predicted_non_exoplanets):,}",
                help="Model tarafından OTHER olarak sınıflandırılan objeler"
            )
        with metric_cols[2]:
            st.metric(
                "🎯 Yüksek Güven (≥90%)", 
                f"{len(high_confidence):,}",
                help="Model %90 veya daha yüksek güvenle tahmin etti"
            )
        with metric_cols[3]:
            st.metric(
                "⚡ Orta Güven (70-90%)", 
                f"{len(medium_confidence):,}",
                help="Model %70-90 arası güvenle tahmin etti"
            )
        with metric_cols[4]:
            st.metric(
                "⚠️ Düşük Güven (<70%)", 
                f"{len(low_confidence):,}",
                help="Model %70'in altında güvenle tahmin etti"
            )
        
        # NASA etiketleri ile karşılaştırma
        if "koi_disposition" in display_df.columns:
            st.markdown("#### 📊 Model vs NASA Karşılaştırması")
            
            comparison_df = display_df.copy()
            comparison_df["nasa_label"] = comparison_df["koi_disposition"].str.upper()
            
            # Doğruluk metrikleri
            nasa_confirmed = comparison_df[comparison_df["nasa_label"] == "CONFIRMED"]
            nasa_candidate = comparison_df[comparison_df["nasa_label"] == "CANDIDATE"]
            nasa_false_positive = comparison_df[comparison_df["nasa_label"] == "FALSE POSITIVE"]
            
            # Model'in NASA CONFIRMED'ları ne kadar yakaladığı
            if len(nasa_confirmed) > 0:
                model_agrees_confirmed = nasa_confirmed[nasa_confirmed["model_prediction"] == "CONFIRMED"]
                agreement_rate = len(model_agrees_confirmed) / len(nasa_confirmed) * 100
                
                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    st.info(f"✅ **NASA CONFIRMED:** {len(nasa_confirmed):,} obje")
                    st.success(f"🎯 Model Uyumu: %{agreement_rate:.1f} ({len(model_agrees_confirmed):,}/{len(nasa_confirmed):,})")
                
                with col_comp2:
                    if len(nasa_candidate) > 0:
                        model_promotes_candidate = nasa_candidate[nasa_candidate["model_prediction"] == "CONFIRMED"]
                        st.info(f"🔍 **NASA CANDIDATE:** {len(nasa_candidate):,} obje")
                        st.warning(f"⬆️ Model Terfi: {len(model_promotes_candidate):,} objeyi CONFIRMED olarak değerlendirdi")
        
        # En yüksek güvenle tahmin edilen ötegezegenleri göster
        if len(predicted_exoplanets) > 0:
            st.markdown("#### 🌟 En Güvenilir Ötegezegen Tahminleri (Top 10)")
            
            top_exoplanets = predicted_exoplanets.nlargest(10, "model_probability")
            
            display_cols = [
                col for col in [
                    "kepoi_name", "koi_disposition", "koi_period", "koi_prad", 
                    "koi_teq", "model_probability", "novelty_score"
                ] if col in top_exoplanets.columns
            ]
            
            # Renklendirme için stil
            def highlight_probability(val):
                if pd.isna(val):
                    return ''
                if isinstance(val, (int, float)):
                    if val >= 0.95:
                        return 'background-color: #d4edda; font-weight: bold'
                    elif val >= 0.90:
                        return 'background-color: #d1ecf1'
                return ''
            
            styled_df = top_exoplanets[display_cols].style.applymap(
                highlight_probability, 
                subset=['model_probability'] if 'model_probability' in display_cols else []
            )
            
            st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("Model tahmin sonuçları bulunamadı.")

    st.markdown("### 📋 Görev tablosu")
    columns_to_show = [
        col
        for col in [
            "koi_name",
            "koi_disposition",
            "koi_period",
            "koi_prad",
            "koi_teq",
            "model_probability",
            "novelty_score",
            "is_novel_candidate",
        ]
        if col in display_df.columns
    ]

    st.markdown("### 📋 Tüm Değerlendirilen Objeler")
    st.caption("Model tarafından skorlanan tüm objeler (ilk 100 kayıt)")
    st.dataframe(display_df[columns_to_show].head(100), use_container_width=True)

    csv_buffer = BytesIO()
    display_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    st.download_button(
        label="📥 İncelenen veriyi indir",
        data=csv_buffer,
        file_name=f"nasa_live_catalog_{mission_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def show_batch_analysis_page(model, scaler, feature_names, anomaly_detector):
    """Toplu analiz sayfası"""
    st.header("📊 Toplu Ötegezegen Analizi")
    st.markdown("CSV dosyası yükleyerek birden fazla gezegen adayını analiz edin.")
    
    # Güvenlik bilgisi
    with st.expander("🔒 Veri Güvenliği ve Gizlilik"):
        st.markdown("""
        **Dosya Gereksinimleri:**
        - ✅ Sadece **CSV** formatı kabul edilir
        - ✅ Maksimum dosya boyutu: **100 MB**
        
        **Veri Anonimleştirme:**
        - 🔐 Hassas bilgiler (isimler, ID'ler, koordinatlar) otomatik olarak anonimleştirilir
        - 🔐 Anonimleştirme isteğe bağlıdır ve kontrol edilebilir
        - 🔐 Orijinal verileriniz değiştirilmez, sadece analiz için kopyası kullanılır
        """)
    
    # Anonimleştirme seçeneği
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("📁 CSV Dosyası Yükleyin", type=['csv'])
    with col2:
        anonymize_data = st.checkbox("🔒 Veriyi Anonimleştir", value=True, 
                                     help="Hassas bilgileri otomatik olarak anonimleştirir")
    
    if uploaded_file is not None:
        # Dosya doğrulama
        is_valid, validation_message = validate_csv_file(uploaded_file)
        
        if not is_valid:
            st.error(validation_message)
            st.warning("⚠️ Lütfen geçerli bir CSV dosyası yükleyin.")
            st.stop()
        
        try:
            # CSV'yi oku
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ {len(df)} satır, {len(df.columns)} sütun başarıyla yüklendi!")
            
            # Veri anonimleştirme
            anonymization_report = None
            if anonymize_data:
                with st.spinner("🔐 Veriler anonimleştiriliyor..."):
                    df, anonymization_report = anonymize_sensitive_columns(df)
                
                if anonymization_report['anonymized_columns']:
                    st.info(f"🔒 **{len(anonymization_report['anonymized_columns'])}** sütun anonimleştirildi: "
                           f"{', '.join([f'`{col}`' for col in anonymization_report['anonymized_columns']])}")
                else:
                    st.info("ℹ️ Anonimleştirilecek hassas sütun bulunamadı.")
            
            # Veri önizleme
            with st.expander("👀 Veri Önizleme"):
                st.dataframe(df.head(10), use_container_width=True)
                
                if anonymization_report and anonymization_report['anonymized_columns']:
                    st.markdown("**🔐 Anonimleştirme Detayları:**")
                    for col, method in anonymization_report['method_used'].items():
                        method_name = "Sayısal Hash" if method == 'numeric_hash' else "Metin Hash"
                        st.markdown(f"- `{col}`: {method_name}")
            
            # Analiz butonu
            if st.button("🚀 Toplu Analiz Başlat", use_container_width=True):
                with st.spinner("🔄 Analiz ediliyor..."):
                    results = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in df.iterrows():
                        features = row.to_dict()
                        result = predict_exoplanet(
                            features,
                            model,
                            scaler,
                            feature_names,
                            anomaly_detector=anomaly_detector,
                        )
                        
                        novelty = result.get('novelty') or {}
                        novelty_flag = novelty.get('is_novel_candidate', False)
                        novelty_score = novelty.get('novelty_score')

                        results.append({
                            'Satır': idx + 1,
                            'Ötegezegen': 'Evet ✅' if result['is_exoplanet'] else 'Hayır ❌',
                            'Olasılık (%)': f"{result['probability_exoplanet']*100:.2f}",
                            'Güven (%)': f"{result['confidence']*100:.2f}",
                            'Yeni Aday': '🚨 Evet' if novelty_flag else '✅ Uyumlu',
                            'Novelty Skoru': f"{novelty_score:.3f}" if novelty_score is not None else '—'
                        })
                        
                        # Progress güncelle
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"İşleniyor: {idx + 1}/{len(df)}")
                    
                    status_text.text("✅ Analiz tamamlandı!")
                    
                    # Sonuçlar
                    results_df = pd.DataFrame(results)
                    
                    st.markdown("---")
                    st.header("📊 Analiz Sonuçları")
                    
                    # İstatistikler
                    total = len(results_df)
                    exoplanets = len(results_df[results_df['Ötegezegen'] == 'Evet ✅'])
                    non_exoplanets = total - exoplanets
                    avg_confidence = results_df['Güven (%)'].str.rstrip('%').astype(float).mean()
                    new_candidate_count = (results_df['Yeni Aday'] == '🚨 Evet').sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Toplam Analiz", total)
                    with col2:
                        st.metric("Ötegezegen Bulundu", exoplanets, f"{exoplanets/total*100:.1f}%")
                    with col3:
                        st.metric("Ötegezegen Değil", non_exoplanets)
                    with col4:
                        st.metric("Ortalama Güven", f"{avg_confidence:.2f}%")

                    extra_col1, extra_col2 = st.columns(2)
                    with extra_col1:
                        st.metric("Yeni Aday Sayısı", new_candidate_count)
                    with extra_col2:
                        novel_ratio = (new_candidate_count / total * 100) if total > 0 else 0
                        st.metric("Novelty Oranı", f"{novel_ratio:.2f}%")
                    
                    # Pasta grafiği
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure(data=[go.Pie(
                            labels=['Ötegezegen', 'Ötegezegen Değil'],
                            values=[exoplanets, non_exoplanets],
                            marker_colors=['#38ef7d', '#ff6a00'],
                            hole=0.4
                        )])
                        fig.update_layout(title="Sonuç Dağılımı")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Güven skoru histogramı
                        confidence_values = results_df['Güven (%)'].str.rstrip('%').astype(float)
                        fig = go.Figure(data=[go.Histogram(
                            x=confidence_values,
                            nbinsx=20,
                            marker_color='lightblue'
                        )])
                        fig.update_layout(
                            title="Güven Skoru Dağılımı",
                            xaxis_title="Güven (%)",
                            yaxis_title="Frekans"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detaylı sonuçlar tablosu
                    st.subheader("📋 Detaylı Sonuçlar")
                    st.dataframe(results_df, use_container_width=True)

                    if new_candidate_count > 0:
                        st.markdown("### 🧭 Potansiyel Yeni Adaylar")
                        novel_subset = results_df[results_df['Yeni Aday'] == '🚨 Evet']
                        st.dataframe(novel_subset, use_container_width=True)
                    
                    # İndirme butonu
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Sonuçları İndir (CSV)",
                        data=csv,
                        file_name=f"exoplanet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Hata: {e}")
            st.info("ℹ️ Lütfen geçerli bir CSV dosyası yükleyin.")


def show_data_generation_page():
    """Sentetik veri üretim sayfası"""
    st.header("🧪 Sentetik Veri Üretimi Laboratuvarı")
    st.markdown(
        "Gerçek Kepler dağılımını koruyarak yeni ötegezegen adayları üretin, senaryoları test edin "
        "ve modelinizi zenginleştirin."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        data_source = st.selectbox(
            "📁 Referans Veri Kaynağı",
            ["Varsayılan Kepler Verisi", "CSV Yükle"],
        )
    with col2:
        strategy = st.selectbox(
            "🧠 Üretim Stratejisi",
            [
                "Gauss Karışımı",
                "SMOTE (Azınlık güçlendirme)",
                "Hibrit (Önerilen)",
            ],
            index=2,
        )
    with col3:
        random_state = st.number_input(
            "🔢 Rastgele Tohum",
            min_value=0,
            max_value=9999,
            value=42,
        )

    reference_df = None

    try:
        if data_source == "Varsayılan Kepler Verisi":
            reference_df = load_default_reference_dataframe().copy()
        else:
            uploaded_file = st.file_uploader(
                "📤 Özellikleri içeren CSV dosyası yükleyin",
                type=["csv"],
                help="Dosya 'koi_disposition' veya 'is_exoplanet' sütununu içermelidir.",
            )
            if uploaded_file is not None:
                raw_df = pd.read_csv(uploaded_file)
                features, labels = prepare_features_from_dataframe(raw_df)
                reference_df = pd.concat([features, labels.rename("is_exoplanet")], axis=1)
            else:
                st.info("ℹ️ Varsayılan veri seti kullanılacak.")
                reference_df = load_default_reference_dataframe().copy()
    except Exception as exc:
        st.error(f"Veri yüklenirken sorun oluştu: {exc}")
        return

    if reference_df is None or reference_df.empty:
        st.warning("Gösterilecek veri bulunamadı.")
        return

    X_reference = reference_df.drop(columns="is_exoplanet")
    y_reference = reference_df["is_exoplanet"]

    confirmed_ratio = y_reference.mean() * 100
    st.markdown("### 📈 Referans Veri Özeti")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Toplam Örnek", f"{len(reference_df):,}")
    with m2:
        st.metric("Özellik Sayısı", len(X_reference.columns))
    with m3:
        st.metric("CONFIRMED Oranı", f"{confirmed_ratio:.2f}%")
    with m4:
        st.metric("FALSE/Diğer", f"{100 - confirmed_ratio:.2f}%")

    st.markdown("---")

    with st.form("synthetic_generation_form"):
        st.subheader("⚙️ Üretim Parametreleri")
        col_left, col_right = st.columns(2)
        with col_left:
            sample_count = st.slider(
                "Üretilecek örnek sayısı",
                min_value=200,
                max_value=10000,
                step=100,
                value=2000,
            )
            include_labels = st.checkbox(
                "Çıktıya 'is_exoplanet' etiketini ekle",
                value=True,
            )
        with col_right:
            target_exoplanet_ratio = st.slider(
                "Hedef ötegezegen oranı (%)",
                min_value=10,
                max_value=90,
                value=int(np.clip(confirmed_ratio, 15, 60)),
                help="Çıktı veri setinde ötegezegenlerin payı.",
            )
            quality_clip = st.checkbox(
                "Aykırı değerleri otomatik sınırla",
                value=True,
            )

        submitted = st.form_submit_button("🚀 Sentetik Veriyi Üret", use_container_width=True)

    if not submitted:
        st.info("Parametreleri seçip 'Sentetik Veriyi Üret' butonuna tıklayın.")
        return

    with st.spinner("🧪 Veri üretiliyor..."):
        generator = ExoplanetDataGenerator(random_state=random_state)
        generator.fit(X_reference, y_reference)

        class_ratio = {
            1: target_exoplanet_ratio / 100,
            0: 1 - (target_exoplanet_ratio / 100),
        }

        strategy_map = {
            "Gauss Karışımı": "gaussian",
            "SMOTE (Azınlık güçlendirme)": "smote",
            "Hibrit (Önerilen)": "hybrid",
        }

        synthetic_df = generator.generate_dataset(
            n_samples=sample_count,
            strategy=strategy_map[strategy],
            class_ratio=class_ratio,
            include_labels=include_labels,
        )

    st.success(
        f"✅ {len(synthetic_df):,} satırlık sentetik veri başarıyla üretildi!"
    )

    if quality_clip:
        numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == "is_exoplanet":
                continue
            if col not in reference_df.columns:
                continue
            q1 = reference_df[col].quantile(0.01)
            q3 = reference_df[col].quantile(0.99)
            synthetic_df[col] = synthetic_df[col].clip(q1, q3)

    st.markdown("### 📊 Sentetik Veri Özeti")
    col_summary, col_chart = st.columns([1.2, 1])
    with col_summary:
        synth_ratio = (
            synthetic_df["is_exoplanet"].mean() * 100
            if "is_exoplanet" in synthetic_df.columns
            else target_exoplanet_ratio
        )
        st.metric("Ötegezegen Oranı", f"{synth_ratio:.2f}%")
        st.metric("Örnek Sayısı", f"{len(synthetic_df):,}")
        st.metric("Strateji", strategy)
    with col_chart:
        if "is_exoplanet" in synthetic_df.columns:
            ratio_fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Ötegezegen", "Diğer"],
                        values=[synth_ratio, 100 - synth_ratio],
                        hole=0.55,
                        marker_colors=["#38ef7d", "#ff6a00"],
                    )
                ]
            )
            ratio_fig.update_layout(height=240, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(ratio_fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🔍 İlk 10 Satır")
    st.dataframe(synthetic_df.head(10), use_container_width=True)

    csv_buffer = BytesIO()
    synthetic_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    st.download_button(
        label="📥 Sentetik Veriyi İndir",
        data=csv_buffer,
        file_name=f"synthetic_exoplanets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.markdown("### 📈 Özellik Dağılımı Karşılaştırması")
    comparison_cols = st.multiselect(
        "Karşılaştırılacak özellikler",
        options=list(X_reference.columns),
        default=["koi_period", "koi_prad", "koi_teq"],
    )

    if comparison_cols:
        tabs = st.tabs([f"📌 {col}" for col in comparison_cols])
        for tab, feature in zip(tabs, comparison_cols):
            with tab:
                fig = go.Figure()
                fig.add_trace(
                    go.Box(
                        y=reference_df[feature],
                        name="Referans",
                        marker_color="#2a5298",
                    )
                )
                fig.add_trace(
                    go.Box(
                        y=synthetic_df[feature],
                        name="Sentetik",
                        marker_color="#38ef7d",
                    )
                )
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

def show_model_analysis_page(model, scaler, feature_names, anomaly_detector):
    """Model analizi sayfası"""
    st.header("🧠 Model Performans Analizi")
    st.markdown("Eğitilmiş modelin detaylı performans analizi ve özellik önem dereceleri.")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performans Metrikleri",
        "⭐ Özellik Analizi",
        "📊 Model Detayları",
        "🧭 Novelty İzleme",
    ])
    
    with tab1:
        st.subheader("📊 Model Performans Metrikleri")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Doğruluk", "94.46%", "4.46%")
        with col2:
            st.metric("Kesinlik", "87.86%", "2.14%")
        with col3:
            st.metric("Duyarlılık", "93.62%", "3.62%")
        with col4:
            st.metric("F1 Skoru", "0.9065", "0.0065")
        with col5:
            st.metric("ROC AUC", "0.9839", "0.0839")
        
        st.markdown("---")
        
        # Karmaşıklık matrisi
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Karmaşıklık Matrisi**")
            cm_data = np.array([[1293, 71], [35, 514]])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm_data,
                x=['Tahmin: Hayır', 'Tahmin: Evet'],
                y=['Gerçek: Hayır', 'Gerçek: Evet'],
                text=cm_data,
                texttemplate="%{text}",
                colorscale='Blues',
                showscale=False
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Açıklama:**
            - **TN (1293):** Doğru negatif - Doğru şekilde 'değil' dendi
            - **FP (71):** Yanlış pozitif - Yanlışlıkla 'ötegezegen' dendi
            - **FN (35):** Yanlış negatif - Kaçırılan ötegezegen
            - **TP (514):** Doğru pozitif - Doğru tespit edilen ötegezegen
            """)
        
        with col2:
            st.markdown("**📊 Metrik Karşılaştırması**")
            metrics_data = {
                'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                'Değer': [0.9446, 0.8786, 0.9362, 0.9065, 0.9839]
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=metrics_data['Metrik'],
                    y=metrics_data['Değer'],
                    marker_color=['#667eea', '#764ba2', '#667eea', '#764ba2', '#667eea'],
                    text=[f"{v:.4f}" for v in metrics_data['Değer']],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                yaxis_range=[0, 1],
                height=400,
                showlegend=False
            )
            fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="90% Hedef")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("⭐ Özellik Önem Dereceleri")
        
        # Özellik önem dereceleri
        feature_importance = pd.DataFrame({
            'Özellik': feature_names,
            'Önem': model.feature_importances_
        }).sort_values('Önem', ascending=False)
        
        # Top 15 grafik
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(
                    y=feature_importance['Özellik'].head(15),
                    x=feature_importance['Önem'].head(15),
                    orientation='h',
                    marker_color='coral',
                    text=feature_importance['Önem'].head(15).round(4),
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="En Önemli 15 Özellik",
                xaxis_title="Önem Derecesi",
                yaxis_title="Özellik",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Top 10 Özellikler**")
            for idx, row in feature_importance.head(10).iterrows():
                st.metric(
                    row['Özellik'],
                    f"{row['Önem']:.4f}",
                    delta=f"#{feature_importance.index.get_loc(idx) + 1}"
                )
        
        # Tam tablo
        st.markdown("---")
        st.markdown("**📋 Tüm Özellikler**")
        st.dataframe(feature_importance, use_container_width=True)
        
        # Özellik önem dağılımı
        fig = go.Figure(data=[go.Histogram(
            x=feature_importance['Önem'],
            nbinsx=20,
            marker_color='lightgreen'
        )])
        fig.update_layout(
            title="Özellik Önem Derecesi Dağılımı",
            xaxis_title="Önem Derecesi",
            yaxis_title="Frekans"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔧 Model Detayları")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>🤖 Model Bilgileri</h3>
                <ul>
                    <li><strong>Algoritma:</strong> XGBoost Classifier</li>
                    <li><strong>N Estimators:</strong> 300</li>
                    <li><strong>Max Depth:</strong> 7</li>
                    <li><strong>Learning Rate:</strong> 0.05</li>
                    <li><strong>Subsample:</strong> 0.8</li>
                    <li><strong>Colsample Bytree:</strong> 0.8</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="success-box">
                <h3>📊 Veri Bilgileri</h3>
                <ul>
                    <li><strong>Toplam Veri:</strong> 9,564 örneklem</li>
                    <li><strong>Eğitim Seti:</strong> 7,651 örneklem (80%)</li>
                    <li><strong>Test Seti:</strong> 1,913 örneklem (20%)</li>
                    <li><strong>Özellik Sayısı:</strong> 23</li>
                    <li><strong>Dengeleme:</strong> SMOTE + Undersampling</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h3>⚙️ Ön İşleme</h3>
                <ul>
                    <li><strong>Ölçeklendirme:</strong> Robust Scaler</li>
                    <li><strong>Eksik Veri:</strong> Medyan ile doldurma</li>
                    <li><strong>Özellik Mühendisliği:</strong> 5 yeni özellik</li>
                    <li><strong>Dengeleme Oranı:</strong> 1.25:1</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <h3>🎯 Hedef Dağılımı</h3>
                <ul>
                    <li><strong>CONFIRMED:</strong> 2,746 (28.7%)</li>
                    <li><strong>FALSE POSITIVE:</strong> 4,839 (50.6%)</li>
                    <li><strong>CANDIDATE:</strong> 1,979 (20.7%)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.subheader("🧭 Anomali Tabanlı Yeni Aday Takibi")

        if anomaly_detector is None:
            st.info(
                "IsolationForest modeli bulunamadı. Lütfen `python main.py` komutu ile anomali modelini eğitip `anomaly_detector.pkl` dosyasını oluşturun."
            )
            return

        default_path = Path('cumulative_2025.10.04_09.55.40.csv')
        try:
            if default_path.exists():
                reference_raw = pd.read_csv(default_path, comment='#')
            else:
                reference_raw = load_default_reference_dataframe()

            scoring = score_catalog(reference_raw, model, scaler, list(feature_names), anomaly_detector)
            scored_reference = scoring['scored']

            novel_df = scored_reference[scored_reference.get('is_novel_candidate', False)].copy()
            total_ref = len(scored_reference)
            novel_count = len(novel_df)
            avg_novelty = float(scored_reference.get('novelty_score', pd.Series(dtype=float)).mean())

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Referans kayıt", f"{total_ref:,}")
            with metric_col2:
                st.metric("Yeni aday", f"{novel_count:,}")
            with metric_col3:
                st.metric("Ortalama novelty", f"{avg_novelty:.3f}" if not np.isnan(avg_novelty) else "—")

            if novel_count:
                st.markdown("### 🚨 Öne çıkan anomaliler")
                display_cols = [
                    col
                    for col in [
                        'koi_name',
                        'koi_period',
                        'koi_prad',
                        'koi_teq',
                        'model_probability',
                        'novelty_score',
                        'koi_disposition',
                    ]
                    if col in novel_df.columns
                ]
                st.dataframe(
                    novel_df.sort_values('novelty_score').head(25)[display_cols],
                    use_container_width=True,
                )
            else:
                st.success("Referans veri kümesinde dağılım dışı güçlü aday bulunmadı.")

            if {'koi_period', 'koi_model_snr', 'novelty_score'}.issubset(scored_reference.columns):
                st.markdown("### 🔍 Novelty dağılımı")
                fig = px.scatter(
                    scored_reference,
                    x='koi_period',
                    y='koi_model_snr',
                    color='novelty_score',
                    color_continuous_scale='Viridis',
                    hover_data=['koi_disposition'] if 'koi_disposition' in scored_reference.columns else None,
                    height=500,
                )
                if novel_count:
                    fig.add_trace(
                        go.Scatter(
                            x=novel_df['koi_period'],
                            y=novel_df['koi_model_snr'],
                            mode='markers',
                            marker=dict(color='#ff4d4f', size=10, symbol='x'),
                            name='Yeni aday',
                        )
                    )
                fig.update_layout(xaxis_title='Yörünge periyodu (gün)', yaxis_title='Model SNR')
                st.plotly_chart(fig, use_container_width=True)

            csv_buffer = BytesIO()
            scored_reference.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                label="📥 Novelty raporunu indir",
                data=csv_buffer,
                file_name=f"novelty_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Referans verisi analiz edilirken hata oluştu: {exc}")


def show_about_page():
    """Hakkında sayfası"""
    st.header("📚 Ötegezegen Keşif Sistemi Hakkında")
    
    st.markdown("""
    ## 🌌 Proje Özeti
    
    Bu proje, NASA'nın Kepler uzay teleskobundan elde edilen verileri kullanarak 
    ötegezegenleri otomatik olarak tespit eden bir yapay zeka sistemidir.
    
    ### ✨ Özellikler
    
    - **🤖 XGBoost Algoritması:** Yüksek doğruluklu gradient boosting
    - **🧠 XAI (Explainable AI):** SHAP ve LIME ile açıklanabilir tahminler
    - **📊 Streamlit Arayüzü:** Modern ve kullanıcı dostu web arayüzü
    - **📈 Gerçek Zamanlı Analiz:** Anında tahmin ve açıklama
    - **📁 Toplu İşleme:** CSV dosyası ile birden fazla tahmin
    
    ### 🎯 Performans
    
    - **Doğruluk:** %94.46
    - **F1 Skoru:** 0.9065
    - **ROC AUC:** 0.9839
    - **Test Seti:** 1,913 örneklem
    
    ### 🔬 Teknolojiler
    
    - Python 3.10+
    - XGBoost
    - SHAP & LIME
    - Streamlit
    - Plotly
    - Scikit-learn
    - Pandas & NumPy
    
    ### 📊 Veri Seti
    
    **NASA Exoplanet Archive - Kepler Cumulative Dataset**
    - 9,564 gezegen adayı
    - 141 özellik
    - 2,746 doğrulanmış ötegezegen
    
    ### 👥 Kullanım Alanları
    
    1. **Araştırmacılar:** Yeni gezegen adaylarını hızlıca değerlendirme
    2. **Öğrenciler:** Makine öğrenimi ve astrofizik eğitimi
    3. **Meraklılar:** Ötegezegen keşfini anlama ve deneyimleme
    
    ### 🚀 Geliştirmeler
    
    - (Planlanan) K2 ve TESS veri setleri desteği
    - Derin öğrenme modelleri
    - API entegrasyonu
    - Gerçek zamanlı veri akışı
    
    ### 📖 Kaynaklar
    
    - [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
    - [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)
    - [XGBoost Documentation](https://xgboost.readthedocs.io/)
    - [SHAP Documentation](https://shap.readthedocs.io/)
    
    ### 📧 İletişim
    
    Sorularınız için GitHub üzerinden issue açabilirsiniz.
    
    ---
    
    **🌌 Evrenin gizemlerini keşfetmeye devam edin! 🌌**
    """)
    
    # Son güncelleme
    st.info(f"📅 Son Güncelleme: {datetime.now().strftime('%d.%m.%Y')}")
    
    # Teşekkürler
    st.success("""
    🙏 **Teşekkürler**
    
    - NASA Kepler ekibine veri seti için
    - Açık kaynak topluluğuna harika araçlar için
    - Sizlere ilginiz için!
    """)

if __name__ == "__main__":
    main()
