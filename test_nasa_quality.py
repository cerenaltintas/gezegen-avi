"""
Basit test - NASA API ve veri kalitesi
"""
import sys
sys.path.insert(0, '.')

from nasa_api import get_latest_dataframe
import pandas as pd

print("=" * 70)
print("NASA API - VERİ KALİTESİ VE MODEL HAZIRLIĞI TESTİ")
print("=" * 70)

try:
    print("\n📡 NASA API'den veri alınıyor (cumulative table)...")
    df = get_latest_dataframe(mission="kepler", limit=200, force_refresh=True)
    print(f"   ✓ {len(df)} kayıt alındı")
    print(f"   ✓ {len(df.columns)} sütun var")
    
    # Kritik sütunları kontrol et
    critical_cols = ['koi_period', 'koi_prad', 'koi_disposition']
    print(f"\n🔍 Kritik Sütun Kontrolü:")
    for col in critical_cols:
        if col in df.columns:
            valid_count = df[col].notna().sum()
            print(f"   ✓ {col}: {valid_count}/{len(df)} geçerli değer")
        else:
            print(f"   ✗ {col}: BULUNAMADI!")
    
    # Disposition dağılımı
    if 'koi_disposition' in df.columns:
        print(f"\n📊 NASA Etiket Dağılımı:")
        disposition_counts = df['koi_disposition'].str.upper().value_counts()
        for disp, count in disposition_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   • {disp}: {count:,} (%{percentage:.1f})")
    
    # 3D görselleştirme için veri kalitesi
    viz_ready = df.dropna(subset=['koi_period', 'koi_prad'])
    print(f"\n🎨 3D Görselleştirme Hazırlığı:")
    print(f"   ✓ {len(viz_ready)}/{len(df)} kayıt görselleştirilebilir")
    print(f"   ✓ Kayıp oran: %{((len(df) - len(viz_ready)) / len(df) * 100):.1f}")
    
    # Örnek veriler
    if len(viz_ready) > 0:
        print(f"\n📋 Örnek Veri (ilk 3 kayıt):")
        sample_cols = ['kepoi_name', 'koi_disposition', 'koi_period', 'koi_prad', 'koi_teq']
        available_sample_cols = [c for c in sample_cols if c in viz_ready.columns]
        print(viz_ready[available_sample_cols].head(3).to_string(index=False))
    
    print(f"\n" + "=" * 70)
    print("✅ NASA API VE VERİ KALİTESİ BAŞARILI!")
    print("=" * 70)
    print("\n💡 Sonraki Adım:")
    print("   Streamlit uygulamasını yeniden başlatın:")
    print("   > python -m streamlit run streamlit_app.py")
    print("\n   Ardından tarayıcıda 'Canlı Veri' sayfasına gidin ve")
    print("   'Önbelleği yenile' kutusunu işaretleyip yenileyin.")
    
except Exception as e:
    print(f"\n✗ HATA: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
