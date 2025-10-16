Yapay Zekâ ile Gezegen Avı projemizde, ötegezegen keşif sürecindeki manuel, zaman alıcı ve hata payı yüksek analizleri otomatikleştirmeyi hedefledik. Bunun için öncelikle proje gerekliliklerini ve kaynaklardaki araştırma projelerini detaylı şekilde analiz ettik. Bu sayede modelimiz, verilen veri kümesindeki nitelikleri açgözlü bir yaklaşımla ağırlıklandırarak, ötegezegen tespitinde daha anlamlı özelliklere öncelik verir hale geldi.

Eğittiğimiz modelin çıktısını kullanıcı dostu hale getirmek için Streamlit tabanlı bir web arayüzü geliştirdik. Araştırmacılar bu arayüz üzerinden:

Basit veya gelişmiş modda ötegezegen tahmini yapabilir,

CSV formatındaki veri kümelerini yükleyerek toplu analiz gerçekleştirebilir,

Varsayılan Kepler verilerinden sentetik veri üretimi yaparak veri tiplerini tanıyabilir,

Ürettiği sentetik verileri kaydedip analiz ederek olası ötegezegen örüntülerini önceden inceleyebilir.

Sistem, ayrıca etkileşimli 3B yıldız sistemi görselleştirmesi sayesinde kullanıcıların tespit ettikleri ötegezegenleri dinamik bir uzay ortamında konumlandırarak görselleştirmelerine olanak tanır. Bu sayede yalnızca sayısal değil, görsel bir keşif deneyimi de sunar.

Veri güvenliği ve doğruluğu açısından, yüklenen veriler üzerinde dosya türü kontrolü yapılır; yalnızca CSV formatı kabul edilir.
Projemizin en özgün bileşenlerinden biri, AI Explainability (Yapay Zekâ Açıklanabilirliği) bileşenidir. Modelin neden belirli bir kararı verdiği, SHAP değerleri aracılığıyla kullanıcıya görsel olarak açıklanır. Böylece araştırmacılar, “Model bu kararı neden verdi?” sorusuna net ve güvenilir bir yanıt alabilir.

Ayrıca sistemimiz, kullanıcı etkileşiminden beslenen bir Active Learning (etkin öğrenme) yaklaşımı da sunar. Kullanıcıların yüklediği veya etiketlediği veriler, zamanla modelin performansını iyileştirmek amacıyla öğrenme sürecine dahil edilebilir.

Son olarak, NASA’nın Exoplanet Archive API adresi üzerinden oluşturduğumuz canlı veri akış sistemi, araştırmacıların en güncel keşifleri anlık olarak takip etmelerini sağlar. Bu API yapısı, yalnızca veri çekmekle kalmaz, aynı zamanda dinamik olarak işlenebilir ve görselleştirilebilir bir akış mantığı sunarak yenilikçi bir yaklaşım sergiler.
