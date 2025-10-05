"""NASA API veri akışını test et"""
import sys
sys.path.insert(0, '.')

from nasa_api import fetch_latest_catalog, get_latest_dataframe
import pandas as pd

print("=" * 70)
print("NASA API VERİ AKIŞ TESTİ")
print("=" * 70)

try:
    print("\n1️⃣ fetch_latest_catalog çağrılıyor...")
    df_raw = fetch_latest_catalog(mission="kepler", limit=500, force_refresh=True)
    print(f"   ✓ Raw veri alındı: {len(df_raw)} kayıt")
    print(f"   ✓ Sütun sayısı: {len(df_raw.columns)}")
    print(f"   ✓ İlk 5 sütun: {list(df_raw.columns[:5])}")
    
    # Büyük/küçük harf kontrolü
    period_cols = [c for c in df_raw.columns if 'period' in c.lower()]
    prad_cols = [c for c in df_raw.columns if 'prad' in c.lower()]
    print(f"   ✓ Period sütunları: {period_cols}")
    print(f"   ✓ Prad sütunları: {prad_cols}")
    
except Exception as e:
    print(f"   ✗ HATA: {e}")
    sys.exit(1)

print("\n2️⃣ get_latest_dataframe çağrılıyor...")
try:
    df = get_latest_dataframe(mission="kepler", limit=500, force_refresh=True)
    print(f"   ✓ İşlenmiş veri alındı: {len(df)} kayıt")
    print(f"   ✓ Sütunlar küçük harfe çevrildi")
    
    # Gerekli sütunları kontrol et
    required = ['koi_period', 'koi_prad']
    for col in required:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"   ✓ {col}: mevcut ({non_null} geçerli değer)")
        else:
            print(f"   ✗ {col}: EKSIK!")
    
except Exception as e:
    print(f"   ✗ HATA: {e}")
    sys.exit(1)

print("\n3️⃣ Veri kalitesi kontrolü...")
minimal_plot_cols = ["koi_period", "koi_prad"]

if all(col in df.columns for col in minimal_plot_cols):
    plot_df = df.dropna(subset=minimal_plot_cols).copy()
    print(f"   ✓ NaN filtreleme öncesi: {len(df)} kayıt")
    print(f"   ✓ NaN filtreleme sonrası: {len(plot_df)} kayıt")
    
    # Sayısal dönüşüm
    for col in minimal_plot_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    
    plot_df = plot_df.dropna(subset=minimal_plot_cols)
    print(f"   ✓ Sayısal dönüşüm sonrası: {len(plot_df)} kayıt")
    
    if plot_df.empty:
        print("   ✗ VERİ BOŞ! Tüm kayıtlar filtrelendi.")
    else:
        print(f"\n   ✅ BAŞARI! {len(plot_df)} kayıt 3D görselleştirme için hazır")
        print(f"\n   Örnek veriler:")
        print(plot_df[['koi_period', 'koi_prad']].head(3).to_string())
else:
    print(f"   ✗ Gerekli sütunlar eksik!")
    print(f"   Mevcut sütunlar: {[c for c in df.columns if 'koi' in c][:10]}")

print("\n" + "=" * 70)
print("TEST TAMAMLANDI")
print("=" * 70)
