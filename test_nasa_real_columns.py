"""NASA Table'ın gerçek sütunlarını keşfet"""
import requests
import io
import pandas as pd

BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# En basit sorgu - sadece ilk 5 kayıt, tüm sütunlar
query = "SELECT TOP 5 * FROM q1_q17_dr25_sup_koi WHERE koi_disposition IS NOT NULL ORDER BY koi_score DESC"

params = {
    "query": query,
    "format": "csv"
}

print("=" * 70)
print("NASA TAP API - GERÇEK SÜTUN KEŞFİ")
print("=" * 70)
print(f"\nSorgu: {query}\n")

response = requests.get(BASE_URL, params=params, timeout=30)

if response.status_code == 200:
    df = pd.read_csv(io.StringIO(response.text))
    print(f"✓ Başarılı! {len(df)} kayıt, {len(df.columns)} sütun döndü\n")
    
    print("TÜM SÜTUNLAR:")
    for i, col in enumerate(df.columns, 1):
        # İlk kayıttaki değeri göster
        sample_val = df[col].iloc[0] if len(df) > 0 else None
        has_data = df[col].notna().sum()
        print(f"  {i:3d}. {col:30s} - {has_data}/5 geçerli - örnek: {sample_val}")
    
    print(f"\n" + "=" * 70)
    print("İLK KAYIT:")
    print("=" * 70)
    print(df.head(1).T)  # Transpose ederek oku
    
else:
    print(f"✗ HATA {response.status_code}:")
    print(response.text[:500])
