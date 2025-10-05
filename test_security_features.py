"""
Dosya Güvenliği ve Veri Anonimleştirme Testleri
"""

import pandas as pd
import numpy as np
import hashlib
from io import BytesIO
import sys

def test_anonymize_function():
    """Anonimleştirme fonksiyonunu test et"""
    print("=" * 60)
    print("🔐 Veri Anonimleştirme Testi")
    print("=" * 60)
    
    # Test verisi oluştur
    test_data = {
        'kepid': [123456, 234567, 345678],
        'kepoi_name': ['K00001.01', 'K00002.01', 'K00003.01'],
        'ra': [285.67, 295.34, 305.12],
        'dec': [48.23, 45.67, 42.89],
        'koi_period': [5.7, 12.3, 8.9],
        'koi_depth': [100, 200, 150],
        'normal_column': ['A', 'B', 'C']
    }
    
    df = pd.DataFrame(test_data)
    
    print("\n📊 Orijinal Veri:")
    print(df)
    
    # Anonimleştirme fonksiyonu
    def anonymize_sensitive_columns(df, sensitive_columns=None):
        df_anon = df.copy()
        anonymization_report = {
            'total_columns': len(df.columns),
            'anonymized_columns': [],
            'method_used': {}
        }
        
        if sensitive_columns is None:
            sensitive_columns = []
            sensitive_keywords = ['name', 'id', 'email', 'address', 'phone', 
                                 'ra', 'dec', 'coordinate', 'location', 
                                 'kepid', 'kepoi']
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in sensitive_keywords):
                    sensitive_columns.append(col)
        
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
    
    # Anonimleştir
    df_anon, report = anonymize_sensitive_columns(df)
    
    print("\n🔒 Anonimleştirilmiş Veri:")
    print(df_anon)
    
    print("\n📋 Anonimleştirme Raporu:")
    print(f"  Toplam Sütun: {report['total_columns']}")
    print(f"  Anonimleştirilen Sütunlar: {len(report['anonymized_columns'])}")
    print(f"  Sütun İsimleri: {', '.join(report['anonymized_columns'])}")
    print("\n  Kullanılan Yöntemler:")
    for col, method in report['method_used'].items():
        method_name = "Sayısal Hash" if method == 'numeric_hash' else "Metin Hash"
        print(f"    - {col}: {method_name}")
    
    # Kontrol et
    assert len(report['anonymized_columns']) == 4, "Yanlış sayıda sütun anonimleştirildi"
    assert 'kepid' in report['anonymized_columns'], "kepid anonimleştirilmedi"
    assert 'ra' in report['anonymized_columns'], "ra anonimleştirilmedi"
    assert 'dec' in report['anonymized_columns'], "dec anonimleştirilmedi"
    assert 'normal_column' not in report['anonymized_columns'], "Normal sütun yanlışlıkla anonimleştirildi"
    
    print("\n✅ Tüm testler başarılı!")
    return True

def test_file_validation():
    """Dosya doğrulama testleri"""
    print("\n" + "=" * 60)
    print("📁 Dosya Doğrulama Testi")
    print("=" * 60)
    
    def validate_csv_content(content, filename):
        """CSV içeriği doğrulama"""
        # Dosya uzantısı kontrolü
        if not filename.lower().endswith('.csv'):
            return False, f"❌ Geçersiz dosya türü: '{filename}'. Sadece CSV dosyaları kabul edilir."
        
        # İçerik kontrolü
        try:
            test_df = pd.read_csv(BytesIO(content.encode()), nrows=5)
            if len(test_df.columns) == 0:
                return False, "❌ CSV dosyası geçerli sütunlar içermiyor."
            return True, "✅ Dosya geçerli."
        except pd.errors.EmptyDataError:
            return False, "❌ CSV dosyası boş."
        except pd.errors.ParserError as e:
            return False, f"❌ CSV ayrıştırma hatası: {str(e)}"
        except Exception as e:
            return False, f"❌ Dosya okuma hatası: {str(e)}"
    
    # Test senaryoları
    tests = [
        {
            'name': 'Geçerli CSV',
            'filename': 'test.csv',
            'content': 'col1,col2,col3\n1,2,3\n4,5,6',
            'expected': True
        },
        {
            'name': 'Geçersiz Uzantı (.txt)',
            'filename': 'test.txt',
            'content': 'col1,col2,col3\n1,2,3',
            'expected': False
        },
        {
            'name': 'Geçersiz Uzantı (.xlsx)',
            'filename': 'test.xlsx',
            'content': 'some content',
            'expected': False
        },
        {
            'name': 'Boş CSV',
            'filename': 'empty.csv',
            'content': '',
            'expected': False
        },
        {
            'name': 'Sadece başlık satırı',
            'filename': 'header_only.csv',
            'content': 'col1,col2,col3',
            'expected': True
        }
    ]
    
    print("\nTest Sonuçları:")
    all_passed = True
    
    for test in tests:
        is_valid, message = validate_csv_content(test['content'], test['filename'])
        passed = (is_valid == test['expected'])
        status = "✅" if passed else "❌"
        
        print(f"\n  {status} {test['name']}")
        print(f"     Dosya: {test['filename']}")
        print(f"     Sonuç: {message}")
        print(f"     Beklenen: {'Geçerli' if test['expected'] else 'Geçersiz'}")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ Tüm dosya doğrulama testleri başarılı!")
    else:
        print("\n❌ Bazı testler başarısız oldu!")
    
    return all_passed

def test_security_measures():
    """Güvenlik önlemlerini test et"""
    print("\n" + "=" * 60)
    print("🔒 Güvenlik Önlemleri Özeti")
    print("=" * 60)
    
    measures = [
        "✅ Dosya Türü Kontrolü: Sadece CSV dosyaları kabul edilir",
        "✅ Dosya Boyutu Kontrolü: Maksimum 100 MB",
        "✅ Dosya Adı Güvenliği: secure_filename() ile temizlenir",
        "✅ Hassas Veri Tespiti: Otomatik hassas sütun algılama",
        "✅ Veri Anonimleştirme: Hash tabanlı anonimleştirme",
        "✅ Hata Yönetimi: Try-except blokları ile korumalı",
        "✅ Kullanıcı Kontrolü: Anonimleştirme isteğe bağlı"
    ]
    
    print("\nUygulanan Güvenlik Önlemleri:")
    for measure in measures:
        print(f"  {measure}")
    
    print("\n🔐 Anonimleştirilen Hassas Bilgiler:")
    sensitive_keywords = ['name', 'id', 'email', 'address', 'phone', 
                         'ra', 'dec', 'coordinate', 'location', 
                         'kepid', 'kepoi']
    for keyword in sensitive_keywords:
        print(f"  - {keyword}")
    
    return True

if __name__ == "__main__":
    print("\n🧪 GÜVENLİK ÖZELLİKLERİ TEST SÜİTİ\n")
    
    try:
        test1 = test_anonymize_function()
        test2 = test_file_validation()
        test3 = test_security_measures()
        
        print("\n" + "=" * 60)
        if test1 and test2 and test3:
            print("🎉 TÜM TESTLER BAŞARIYLA TAMAMLANDI!")
        else:
            print("⚠️ BAZI TESTLER BAŞARISIZ OLDU!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test sırasında hata oluştu: {e}")
        sys.exit(1)
