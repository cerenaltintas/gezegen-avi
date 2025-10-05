"""
Veri Analizi ve Görselleştirme
Kepler veri setinin detaylı analizi ve görselleştirmesi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(file_path):
    """Veriyi yükle"""
    print("📂 Veri yükleniyor...")
    df = pd.read_csv(file_path, comment='#')
    print(f"✅ {len(df)} satır, {len(df.columns)} sütun yüklendi\n")
    return df

def analyze_target_distribution(df):
    """Hedef değişken dağılımını analiz et"""
    print("=" * 80)
    print("HEDEF DEĞİŞKEN ANALİZİ")
    print("=" * 80)
    
    if 'koi_disposition' not in df.columns:
        print("❌ koi_disposition sütunu bulunamadı!")
        return
    
    counts = df['koi_disposition'].value_counts()
    percentages = df['koi_disposition'].value_counts(normalize=True) * 100
    
    print("\n📊 Disposition Dağılımı:")
    for label, count in counts.items():
        pct = percentages[label]
        print(f"  {label:20s}: {count:6d} ({pct:5.2f}%)")
    
    print(f"\n🎯 Toplam CONFIRMED ötegezegen: {counts.get('CONFIRMED', 0)}")
    print(f"❌ Toplam FALSE POSITIVE: {counts.get('FALSE POSITIVE', 0)}")
    print(f"❓ Toplam CANDIDATE: {counts.get('CANDIDATE', 0)}")
    
    # Dengesizlik oranı
    confirmed = counts.get('CONFIRMED', 0)
    total_others = len(df) - confirmed
    if confirmed > 0:
        ratio = total_others / confirmed
        print(f"\n⚖️  Dengesizlik oranı: {ratio:.2f}:1")

def analyze_missing_data(df):
    """Eksik veri analizi"""
    print("\n" + "=" * 80)
    print("EKSİK VERİ ANALİZİ")
    print("=" * 80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Sütun': missing.index,
        'Eksik Sayı': missing.values,
        'Eksik %': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Eksik Sayı'] > 0].sort_values('Eksik %', ascending=False)
    
    print(f"\n📊 Eksik veri içeren sütun sayısı: {len(missing_df)}")
    
    if len(missing_df) > 0:
        print("\n🔝 En çok eksik veri içeren 15 sütun:")
        print(missing_df.head(15).to_string(index=False))
    else:
        print("\n✅ Hiç eksik veri yok!")

def analyze_key_features(df):
    """Ana özelliklerin istatistiksel analizi"""
    print("\n" + "=" * 80)
    print("ANA ÖZELLİKLER İSTATİSTİKSEL ANALİZ")
    print("=" * 80)
    
    key_features = [
        'koi_period',
        'koi_depth',
        'koi_duration',
        'koi_prad',
        'koi_teq',
        'koi_insol',
        'koi_steff',
        'koi_srad'
    ]
    
    available_features = [f for f in key_features if f in df.columns]
    
    if len(available_features) == 0:
        print("❌ Ana özellikler bulunamadı!")
        return
    
    print("\n📊 Özellik İstatistikleri:\n")
    
    stats_df = df[available_features].describe().T
    stats_df['missing_%'] = (df[available_features].isnull().sum() / len(df) * 100).values
    
    print(stats_df.to_string())

def create_visualizations(df):
    """Görselleştirmeler oluştur"""
    print("\n" + "=" * 80)
    print("GÖRSELLEŞTİRMELER OLUŞTURULUYOR")
    print("=" * 80)
    
    # Figure oluştur
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Disposition dağılımı
    ax1 = plt.subplot(2, 3, 1)
    if 'koi_disposition' in df.columns:
        counts = df['koi_disposition'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        ax1.pie(counts.values, labels=counts.index, autopct='%1.1f%%', 
                startangle=90, colors=colors[:len(counts)])
        ax1.set_title('Disposition Dağılımı', fontsize=14, fontweight='bold')
    
    # 2. Yörünge periyodu dağılımı
    ax2 = plt.subplot(2, 3, 2)
    if 'koi_period' in df.columns and 'koi_disposition' in df.columns:
        for disp in df['koi_disposition'].unique():
            data = df[df['koi_disposition'] == disp]['koi_period'].dropna()
            if len(data) > 0:
                ax2.hist(np.log10(data), bins=50, alpha=0.6, label=disp)
        ax2.set_xlabel('log10(Yörünge Periyodu - gün)')
        ax2.set_ylabel('Frekans')
        ax2.set_title('Yörünge Periyodu Dağılımı', fontsize=14, fontweight='bold')
        ax2.legend()
    
    # 3. Gezegen yarıçapı dağılımı
    ax3 = plt.subplot(2, 3, 3)
    if 'koi_prad' in df.columns:
        prad_data = df['koi_prad'].dropna()
        if len(prad_data) > 0:
            ax3.hist(prad_data[prad_data <= 20], bins=50, color='skyblue', edgecolor='black')
            ax3.axvline(x=1, color='r', linestyle='--', label='Dünya Yarıçapı')
            ax3.set_xlabel('Gezegen Yarıçapı (Dünya yarıçapı)')
            ax3.set_ylabel('Frekans')
            ax3.set_title('Gezegen Yarıçapı Dağılımı', fontsize=14, fontweight='bold')
            ax3.legend()
    
    # 4. Geçiş derinliği vs Gezegen yarıçapı
    ax4 = plt.subplot(2, 3, 4)
    if 'koi_depth' in df.columns and 'koi_prad' in df.columns and 'koi_disposition' in df.columns:
        for disp in ['CONFIRMED', 'FALSE POSITIVE']:
            if disp in df['koi_disposition'].values:
                data = df[df['koi_disposition'] == disp]
                ax4.scatter(data['koi_depth'], data['koi_prad'], 
                          alpha=0.5, s=20, label=disp)
        ax4.set_xlabel('Geçiş Derinliği (ppm)')
        ax4.set_ylabel('Gezegen Yarıçapı (Dünya yarıçapı)')
        ax4.set_title('Geçiş Derinliği vs Gezegen Yarıçapı', fontsize=14, fontweight='bold')
        ax4.set_xlim(0, 10000)
        ax4.set_ylim(0, 20)
        ax4.legend()
    
    # 5. Yıldız sıcaklığı dağılımı
    ax5 = plt.subplot(2, 3, 5)
    if 'koi_steff' in df.columns:
        steff_data = df['koi_steff'].dropna()
        if len(steff_data) > 0:
            ax5.hist(steff_data, bins=50, color='orange', edgecolor='black')
            ax5.axvline(x=5778, color='r', linestyle='--', label='Güneş Sıcaklığı')
            ax5.set_xlabel('Yıldız Sıcaklığı (K)')
            ax5.set_ylabel('Frekans')
            ax5.set_title('Yıldız Sıcaklığı Dağılımı', fontsize=14, fontweight='bold')
            ax5.legend()
    
    # 6. Eksik veri ısı haritası
    ax6 = plt.subplot(2, 3, 6)
    important_cols = ['koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 
                     'koi_teq', 'koi_steff', 'koi_srad', 'koi_model_snr']
    available_cols = [c for c in important_cols if c in df.columns]
    
    if len(available_cols) > 0:
        missing_matrix = df[available_cols].isnull().astype(int)
        sns.heatmap(missing_matrix.corr(), annot=True, fmt='.2f', 
                   cmap='RdYlGn_r', ax=ax6, cbar_kws={'label': 'Korelasyon'})
        ax6.set_title('Eksik Veri Korelasyonu', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Görselleştirmeler 'data_analysis.png' olarak kaydedildi!")
    plt.show()

def create_correlation_heatmap(df):
    """Özellikler arası korelasyon ısı haritası"""
    print("\n📊 Korelasyon ısı haritası oluşturuluyor...")
    
    numeric_features = [
        'koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 
        'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad', 
        'koi_model_snr', 'koi_impact'
    ]
    
    available_features = [f for f in numeric_features if f in df.columns]
    
    if len(available_features) < 2:
        print("❌ Yeterli sayıda özellik bulunamadı!")
        return
    
    # Korelasyon matrisi
    corr_matrix = df[available_features].corr()
    
    # Görselleştirme
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={'label': 'Korelasyon'})
    plt.title('Özellikler Arası Korelasyon Matrisi', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Korelasyon ısı haritası 'correlation_heatmap.png' olarak kaydedildi!")
    plt.show()

def analyze_confirmed_vs_false_positive(df):
    """CONFIRMED ve FALSE POSITIVE gezegenleri karşılaştır"""
    print("\n" + "=" * 80)
    print("CONFIRMED vs FALSE POSITIVE KARŞILAŞTIRMASI")
    print("=" * 80)
    
    if 'koi_disposition' not in df.columns:
        return
    
    confirmed = df[df['koi_disposition'] == 'CONFIRMED']
    false_pos = df[df['koi_disposition'] == 'FALSE POSITIVE']
    
    features_to_compare = [
        ('koi_period', 'Yörünge Periyodu (gün)'),
        ('koi_depth', 'Geçiş Derinliği (ppm)'),
        ('koi_duration', 'Geçiş Süresi (saat)'),
        ('koi_prad', 'Gezegen Yarıçapı (Dünya yarıçapı)'),
        ('koi_model_snr', 'Sinyal-Gürültü Oranı')
    ]
    
    print("\n📊 Ortalama Değer Karşılaştırması:\n")
    print(f"{'Özellik':<35} {'CONFIRMED':>15} {'FALSE POSITIVE':>15} {'Fark':>15}")
    print("-" * 80)
    
    for feature, label in features_to_compare:
        if feature in df.columns:
            conf_mean = confirmed[feature].mean()
            fp_mean = false_pos[feature].mean()
            diff = conf_mean - fp_mean
            
            print(f"{label:<35} {conf_mean:>15.2f} {fp_mean:>15.2f} {diff:>15.2f}")

def generate_report(df):
    """Detaylı rapor oluştur"""
    print("\n" + "=" * 80)
    print("DETAYLI VERİ RAPORU")
    print("=" * 80)
    
    report = []
    report.append("\n🌌 KEPLER VERİ SETİ ANALİZ RAPORU\n")
    report.append("=" * 80 + "\n")
    
    # Genel bilgiler
    report.append(f"📊 Toplam Veri Noktası: {len(df):,}\n")
    report.append(f"📊 Toplam Özellik Sayısı: {len(df.columns)}\n")
    report.append(f"📊 Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    # Disposition özeti
    if 'koi_disposition' in df.columns:
        report.append("\n🎯 Disposition Özeti:\n")
        for label, count in df['koi_disposition'].value_counts().items():
            pct = (count / len(df)) * 100
            report.append(f"  {label}: {count:,} ({pct:.2f}%)\n")
    
    # Veri kalitesi
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells) * 100
    
    report.append(f"\n📊 Veri Kalitesi:\n")
    report.append(f"  Toplam eksik değer: {total_missing:,} ({missing_pct:.2f}%)\n")
    report.append(f"  Tam dolu sütunlar: {(df.isnull().sum() == 0).sum()}\n")
    report.append(f"  Eksik veri içeren sütunlar: {(df.isnull().sum() > 0).sum()}\n")
    
    # Sayısal özellikler özeti
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report.append(f"\n📊 Sayısal Özellikler: {len(numeric_cols)}\n")
    
    # Raporu kaydet
    report_text = "".join(report)
    
    with open('data_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print("\n💾 Detaylı rapor 'data_report.txt' olarak kaydedildi!")

def main():
    """Ana fonksiyon"""
    print("\n" + "=" * 80)
    print("🌌 KEPLER VERİ SETİ ANALİZİ VE GÖRSELLEŞTİRME 🌌")
    print("=" * 80 + "\n")
    
    # Veri dosyası yolu
    data_path = 'cumulative_2025.10.04_09.55.40.csv'
    
    if not Path(data_path).exists():
        print(f"❌ Veri dosyası bulunamadı: {data_path}")
        return
    
    # Veriyi yükle
    df = load_data(data_path)
    
    # Analizler
    analyze_target_distribution(df)
    analyze_missing_data(df)
    analyze_key_features(df)
    analyze_confirmed_vs_false_positive(df)
    
    # Görselleştirmeler
    create_visualizations(df)
    create_correlation_heatmap(df)
    
    # Rapor oluştur
    generate_report(df)
    
    print("\n" + "=" * 80)
    print("✨ Analiz tamamlandı! ✨")
    print("=" * 80 + "\n")
    print("📄 Oluşturulan dosyalar:")
    print("  - data_analysis.png (Ana görselleştirmeler)")
    print("  - correlation_heatmap.png (Korelasyon ısı haritası)")
    print("  - data_report.txt (Detaylı rapor)")
    print("")

if __name__ == "__main__":
    main()
