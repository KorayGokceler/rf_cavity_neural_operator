<!-- https://docs.truba.gov.tr/1-kaynaklar/arf_ai_ent/kullanim.html ; modified 2026-04-17 -->

# NVIDIA AI Enterprise Kullanımı

Bu bölüm, ARF-AI üzerinde merkezi olarak sunulan NVIDIA AI Enterprise SIF imajlarının Apptainer ile nasıl çalıştırılacağını açıklar.

## Lisanslı NVIDIA AI Enterprise SIF Konteynerleri

ARF-AI platformunda çeşitli NVIDIA AI Enterprise konteynerleri lisanslı olarak aşağıdaki dizinde sunulmaktadır:

```
/arf/ai-ent/apptainer/sif/nvidia
```

Sık kullanılan bazı örnek konteynerler:

* `nemo-26.02.sif`
* `pytorch-26.03-py3.sif`
* `sglang-26.03-py3.sif`
* `tensorflow-25.02-tf2-py3.sif`
* `tensorrt-26.03-py3.sif`
* `vllm-26.03-py3.sif`
* `ai-dynamo/sglang-runtime-1.1.0-dev.2.sif`

Daha güncel veya tam listeyi görmek için doğrudan ilgili dizinde aşağıdaki gibi komutları kullanabilirsiniz:

```
ls -lh /arf/ai-ent/apptainer/sif/nvidia
ls -lh /arf/ai-ent/apptainer/sif/nvidia/ai-dynamo
```

Ayrıca bakınız

Apptainer temel kavramları için [Konteyner Kullanımı](../../2-temel_bilgiler/konteyner_kullanimi.html#apptainer) sayfasını inceleyiniz.

## İhtiyaç Duyulan Konteyner Bildirimi

Listede bulunmayan bir NVIDIA AI Enterprise konteynerine ihtiyaç duyuyorsanız, aşağıdaki bilgilerle talep iletebilirsiniz:

* Talep edilen konteyner adı ve sürümü ([NGC Kataloğu](https://catalog.ngc.nvidia.com/) üzerinden öğrenilebilir)
* Kullanım amacı (eğitim, çıkarım, servis vb.)
* Gerekli ek paketler veya bağımlılıklar
* Hedef kuyruk/GPU gereksinimi

Talep ve destek için [Teknik Destek Talebi](../../0-truba/truba_iletisim.html#truba-iletisim) sayfasındaki iletişim kanallarını kullanınız.
