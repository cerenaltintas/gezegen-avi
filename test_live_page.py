import pandas as pd
import numpy as np

# Veriyi yükle
df_raw = pd.read_csv('cumulative_2025.10.04_09.55.40.csv', comment='#')
display_df = df_raw.copy()

print("=" * 60)
print("CANLI VERİ SAYFASI TEST")
print("=" * 60)

# 3D görselleştirme için minimum gereksinimler
minimal_plot_cols = ["koi_period", "koi_prad"]
optional_plot_col = "koi_teq"

if all(col in display_df.columns for col in minimal_plot_cols) and not display_df.empty:
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
            plot_df[optional_plot_col] = 300
        else:
            plot_df[optional_plot_col] = plot_df[optional_plot_col].fillna(
                plot_df[optional_plot_col].median()
            )
    else:
        plot_df[optional_plot_col] = 300
    
    print(f"✓ 3D görselleştirme için hazır kayıt sayısı: {len(plot_df):,}")
    print(f"  - koi_period geçerli: {plot_df['koi_period'].notna().sum():,}")
    print(f"  - koi_prad geçerli: {plot_df['koi_prad'].notna().sum():,}")
    print(f"  - koi_teq geçerli: {plot_df['koi_teq'].notna().sum():,}")
else:
    print("✗ 3D görselleştirme için gerekli sütunlar bulunamadı")

print("\n" + "=" * 60)
print("YAŞANABİLİR BÖLGE ANALİZİ")
print("=" * 60)

# Yaşanabilir bölge hesaplaması
# Model probability'yi simüle et
display_df['model_probability'] = np.random.uniform(0.3, 0.95, len(display_df))

hz_candidates = pd.DataFrame()
if all(col in display_df.columns for col in ["koi_prad", "model_probability"]):
    hz_df = display_df.copy()
    
    if "koi_teq" in hz_df.columns:
        hz_df["koi_teq"] = pd.to_numeric(hz_df["koi_teq"], errors="coerce")
        if not hz_df["koi_teq"].isna().all():
            hz_df["koi_teq"] = hz_df["koi_teq"].fillna(hz_df["koi_teq"].median())
        else:
            hz_df["koi_teq"] = 300
    else:
        hz_df["koi_teq"] = 300
    
    # Yaşanabilir bölge filtreleme
    hz_candidates = hz_df[
        hz_df["koi_teq"].between(180, 320)
        & hz_df["koi_prad"].between(0.5, 2.5)
        & (hz_df["model_probability"] >= 0.7)
    ]
    
    print(f"✓ Yaşanabilir bölge adayları: {len(hz_candidates):,}")
    print(f"  - Sıcaklık aralığı (180-320K): {hz_df['koi_teq'].between(180, 320).sum():,}")
    print(f"  - Yarıçap aralığı (0.5-2.5 R⊕): {hz_df['koi_prad'].between(0.5, 2.5).sum():,}")
    print(f"  - Yüksek olasılık (≥0.7): {(hz_df['model_probability'] >= 0.7).sum():,}")
    
    if len(hz_candidates) > 0:
        print(f"\nÖrnek yaşanabilir bölge adayları:")
        sample_cols = ['koi_teq', 'koi_prad', 'model_probability']
        print(hz_candidates[sample_cols].head(3).to_string())
else:
    print("✗ Yaşanabilir bölge analizi için gerekli sütunlar bulunamadı")

print("\n" + "=" * 60)
print("TEST TAMAMLANDI")
print("=" * 60)
