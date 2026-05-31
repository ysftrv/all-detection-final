"""
explore.py — Veri setini keşfet; sınıf dağılımı, boyut istatistikleri ve örnek
             görüntüler içeren bir HTML/Markdown raporu üret.

Kullanım:
    python -m src.data.explore

Çıktılar (experiments/figures/ ve experiments/tables/ altına kaydedilir):
    - class_distribution.png  : ALL vs Normal örnek sayıları (bar chart)
    - image_size_stats.csv    : min/max/mean H ve W
    - sample_grid.png         : her sınıftan 8 örnek görüntü ızgarası
    - explore_report.md       : tüm bulgular özet metin
"""

# TODO (Faz 1):
#   1. config.yaml'ı yükle; data_raw yolunu al.
#   2. Her fold/split klasörü altındaki ALL ve HEM görüntülerini listele.
#   3. Toplam örnek sayısını, sınıf başına sayıyı ve oranı hesapla.
#   4. Görüntülerin piksel boyutlarını tara → min/max/mean/std raporla.
#   5. matplotlib ile class_distribution.png oluştur.
#   6. Her sınıftan 8 rastgele örneği yan yana gösteren sample_grid.png kaydet.
#   7. Bulgular + veri seti lisans bilgisi (C-NMC 2019 kısıtlamaları) explore_report.md'ye yaz.


def main():
    raise NotImplementedError("TODO (Faz 1): explore.py henüz implemente edilmedi.")


if __name__ == "__main__":
    main()
