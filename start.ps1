# Ötegezegen Keşif Sistemi - Başlangıç Scripti
# Bu script projeyi başlatmak için gerekli adımları takip eder

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🌌 ÖTEGEZEGEN KEŞİF SİSTEMİ 🌌" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Menü
Write-Host "Lütfen yapmak istediğiniz işlemi seçin:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Bağımlılıkları Yükle" -ForegroundColor Green
Write-Host "2. Modeli Eğit" -ForegroundColor Green
Write-Host "3. Web Arayüzünü Başlat" -ForegroundColor Green
Write-Host "4. Örnek Tahminleri Çalıştır" -ForegroundColor Green
Write-Host "5. Tam Kurulum (Tümünü Çalıştır)" -ForegroundColor Green
Write-Host "6. Çıkış" -ForegroundColor Red
Write-Host ""

$choice = Read-Host "Seçiminiz (1-6)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "📦 Bağımlılıklar yükleniyor..." -ForegroundColor Yellow
        pip install -r requirements.txt
        Write-Host ""
        Write-Host "✅ Bağımlılıklar başarıyla yüklendi!" -ForegroundColor Green
    }
    "2" {
        Write-Host ""
        Write-Host "🎓 Model eğitiliyor..." -ForegroundColor Yellow
        Write-Host "⚠️  Bu işlem birkaç dakika sürebilir..." -ForegroundColor Yellow
        Write-Host ""
        python main.py
        Write-Host ""
        Write-Host "✅ Model eğitimi tamamlandı!" -ForegroundColor Green
    }
    "3" {
        Write-Host ""
        # Model dosyalarının varlığını kontrol et
        if (Test-Path "exoplanet_model.pkl") {
            Write-Host "🚀 Web arayüzü başlatılıyor..." -ForegroundColor Yellow
            Write-Host "📡 Tarayıcınızda http://127.0.0.1:5000 adresini açın" -ForegroundColor Cyan
            Write-Host ""
            python app.py
        } else {
            Write-Host "❌ Model dosyası bulunamadı!" -ForegroundColor Red
            Write-Host "ℹ️  Lütfen önce '2. Modeli Eğit' seçeneğini çalıştırın." -ForegroundColor Yellow
        }
    }
    "4" {
        Write-Host ""
        if (Test-Path "exoplanet_model.pkl") {
            Write-Host "🔮 Örnek tahminler çalıştırılıyor..." -ForegroundColor Yellow
            Write-Host ""
            python predict_example.py
        } else {
            Write-Host "❌ Model dosyası bulunamadı!" -ForegroundColor Red
            Write-Host "ℹ️  Lütfen önce '2. Modeli Eğit' seçeneğini çalıştırın." -ForegroundColor Yellow
        }
    }
    "5" {
        Write-Host ""
        Write-Host "🚀 Tam kurulum başlatılıyor..." -ForegroundColor Yellow
        Write-Host ""
        
        # Adım 1: Bağımlılıkları yükle
        Write-Host "Adım 1/3: Bağımlılıklar yükleniyor..." -ForegroundColor Cyan
        pip install -r requirements.txt
        Write-Host "✅ Bağımlılıklar yüklendi!" -ForegroundColor Green
        Write-Host ""
        
        # Adım 2: Modeli eğit
        Write-Host "Adım 2/3: Model eğitiliyor..." -ForegroundColor Cyan
        Write-Host "⚠️  Bu işlem birkaç dakika sürebilir..." -ForegroundColor Yellow
        Write-Host ""
        python main.py
        Write-Host ""
        Write-Host "✅ Model eğitildi!" -ForegroundColor Green
        Write-Host ""
        
        # Adım 3: Web arayüzünü başlat
        Write-Host "Adım 3/3: Web arayüzü başlatılıyor..." -ForegroundColor Cyan
        Write-Host "📡 Tarayıcınızda http://127.0.0.1:5000 adresini açın" -ForegroundColor Yellow
        Write-Host ""
        python app.py
    }
    "6" {
        Write-Host ""
        Write-Host "👋 Görüşmek üzere!" -ForegroundColor Cyan
        Write-Host ""
        exit
    }
    default {
        Write-Host ""
        Write-Host "❌ Geçersiz seçim!" -ForegroundColor Red
        Write-Host ""
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Programı tekrar çalıştırmak için:" -ForegroundColor Yellow
Write-Host ".\start.ps1" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
