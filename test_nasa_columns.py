"""NASA API'den dönen gerçek sütunları incele"""
import sys
sys.path.insert(0, '.')

from nasa_api import fetch_latest_catalog
import pandas as pd

df = fetch_latest_catalog(mission="kepler", limit=10, force_refresh=True)

print("=" * 70)
print("NASA API SÜTUN ANALİZİ")
print("=" * 70)

print(f"\nToplam sütun sayısı: {len(df.columns)}")
print(f"\nTÜM SÜTUNLAR:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\n" + "=" * 70)
print("İLK 3 KAYIT:")
print("=" * 70)
print(df.head(3).to_string())

print(f"\n" + "=" * 70)
print("KOI_PRAD DETAY ANALİZİ:")
print("=" * 70)
if 'koi_prad' in df.columns:
    print(f"NaN sayısı: {df['koi_prad'].isna().sum()}")
    print(f"Geçerli sayısı: {df['koi_prad'].notna().sum()}")
    print(f"İlk 5 değer: {df['koi_prad'].head().tolist()}")
else:
    print("✗ koi_prad sütunu bulunamadı!")

# Sütun isimlerinde 'rad' içeren tüm sütunlar
rad_cols = [c for c in df.columns if 'rad' in c.lower()]
print(f"\n'rad' içeren sütunlar: {rad_cols}")

period_cols = [c for c in df.columns if 'period' in c.lower()]
print(f"'period' içeren sütunlar: {period_cols}")
