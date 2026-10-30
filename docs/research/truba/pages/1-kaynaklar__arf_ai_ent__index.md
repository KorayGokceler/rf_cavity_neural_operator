<!-- https://docs.truba.gov.tr/1-kaynaklar/arf_ai_ent/index.html ; modified 2026-04-17 -->

# ARF-AI

[Bağlantı](baglanti.html#arf-ai-ent-baglanti)

[NVIDIA AI Enterprise Kullanımı](kullanim.html#arf-ai-ent-kullanim)

TRUBA’da NVIDIA AI Enterprise platformu lisanslı olarak kullanıma sunulmuştur. Detaylar için [NVIDIA AI Enterprise ürün sayfası](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/) ziyaret edilebilir.
NVIDIA AI Enterprise lisans kapsamındaki konteynerler sistem tarafından hazır olarak oluşturulmuş olup Apptainer aracı ile doğrudan kullanılabilmektedir.

## NVIDIA AI Enterprise Nedir?

[NVIDIA AI Enterprise](https://docs.nvidia.com/ai-enterprise/), yapay zeka modellerinin geliştirilmesi, optimize edilmesi ve dağıtılması için tasarlanmış kapsamlı bir yazılım platformudur. Bu platform, araştırmacılara ve geliştiricilere GPU-hızlandırılmış araçlar, önceden optimize edilmiş konteynerler ve hazır AI mikroservisleri sunarak yapay zeka projelerini önemli ölçüde hızlandırır.

## Platformun Sağladığı Temel Bileşenler

**NVIDIA NIM (Inference Microservices)**

Önceden optimize edilmiş, dağıtıma hazır AI model mikroservisleridir. Her NIM konteyneri; Triton Inference Server, TensorRT, TensorRT-LLM ve PyTorch gibi bileşenleri içerir. Büyük dil modelleri (LLM), görüntü üretimi ve bilimsel modeller gibi pek çok alanda kullanıma hazır konteynerler sunar. ([NIM Dokümantasyonu](https://docs.nvidia.com/nim/index.html) | [API Deneme Platformu](https://build.nvidia.com/explore/discover))

**NVIDIA AI Blueprint**

Yapay zeka uygulamalarının hızla geliştirilmesi için sunulan hazır referans iş akışlarıdır. Blueprint’ler; önceden eğitilmiş modelleri, NIM mikroservislerini ve altyapı bileşenlerini bir araya getirerek RAG (Retrieval-Augmented Generation) tabanlı soru-cevap sistemleri, dijital asistanlar ve çok modlu veri işleme gibi uçtan uca çözümler oluşturmayı kolaylaştırır. ([AI Blueprint Dokümantasyonu](https://build.nvidia.com/blueprints) | [GitHub](https://github.com/NVIDIA-AI-Blueprints))

**NGC Kataloğu Erişimi**

NVIDIA GPU Cloud (NGC) kataloğu üzerinden 100’den fazla AI framework’üne ve önceden eğitilmiş modele erişim sağlanır. NGC, yapay zeka ve HPC uygulamaları için optimize edilmiş konteyner imajlarının merkezi deposudur. ([NGC Kataloğu](https://catalog.ngc.nvidia.com/))

**NVIDIA RAPIDS**

GPU-hızlandırılmış veri bilimi kütüphaneleri ailesidir. cuDF (veri çerçeveleri), cuML (makine öğrenmesi) ve cuGraph (graf analitiği) gibi bileşenlerle veri bilimi süreçlerini 70 kata kadar hızlandırabilir. ([RAPIDS Dokümantasyonu](https://docs.rapids.ai/))

**NVIDIA Triton Inference Server**

Birden fazla framework’ten (PyTorch, TensorFlow, TensorRT, ONNX vb.) gelen modelleri yüksek performansla sunmak için tasarlanmış açık kaynaklı bir çıkarım sunucusudur. Dinamik gruplama ve model birleştirme gibi özelliklerle 100 kata kadar düşük gecikme sağlayabilir. ([Triton Dokümantasyonu](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html))

**NVIDIA TensorRT**

Derin öğrenme modellerini GPU üzerinde en yüksek verimlilikle çalıştırmak için optimize eden bir çıkarım motorudur. Eğitilmiş modelleri otomatik olarak optimize ederek çıkarım hızını önemli ölçüde artırır. ([TensorRT Dokümantasyonu](https://docs.nvidia.com/deeplearning/tensorrt/index.html))

**NVIDIA TAO Toolkit**

Transfer öğrenme tabanlı bir model geliştirme aracıdır. Önceden eğitilmiş modelleri kendi veri kümenizle ince ayar yaparak, sıfırdan eğitime kıyasla AI model geliştirme sürecini 10 kata kadar hızlandırır. ([TAO Toolkit Dokümantasyonu](https://docs.nvidia.com/tao/tao-toolkit/index.html))

**NeMo Framework**

Büyük dil modelleri (LLM), konuşma tanıma (ASR) ve metin-konuşma (TTS) modelleri için kapsamlı bir eğitim ve ince ayar çerçevesidir. Çoklu GPU ve çoklu düğüm eğitimi için optimize edilmiştir. ([NeMo Dokümantasyonu](https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html))

**NVIDIA Dynamo**

Büyük dil modellerinin çok düğümlü dağıtık ortamlarda sunulması için tasarlanmış modüler bir çıkarım framework’üdür. Prefill ve decode aşamalarını ayıran ayrıştırılmış servis mimarisi, önek tabanlı akıllı yönlendirme ve dağıtık KV Cache yönetimi ile GPU kaynaklarını dinamik olarak tahsis ederek çıkarım performansını optimize eder. ([Dynamo GitHub](https://github.com/ai-dynamo/dynamo) | [Geliştirici Sayfası](https://developer.nvidia.com/dynamo))

**AIPerf**

Çıkarım sunucularının performansını ölçmek için tasarlanmış kapsamlı bir benchmark aracıdır (GenAI-Perf’in devamı niteliğindedir). NIM, Triton, vLLM ve SGLang gibi çıkarım çözümlerini destekler; token üretim hızı, ilk token süresi (TTFT), gecikme ve eşzamanlı istek kapasitesi gibi metrikleri raporlar. ([AIPerf GitHub](https://github.com/ai-dynamo/aiperf))

**Desteklenen Framework’ler**

NVIDIA AI Enterprise; PyTorch, TensorFlow, TensorRT, RAPIDS, JAX ve daha pek çok popüler yapay zeka framework’ünü destekler. Tüm bu framework’ler NGC üzerinden optimize edilmiş konteynerler olarak sunulur.

Daha fazla bilgi için aşağıdaki başlıklara göz atabilirsiniz:

* [Bağlantı](baglanti.html#arf-ai-ent-baglanti)
* [NVIDIA AI Enterprise Kullanımı](kullanim.html#arf-ai-ent-kullanim)
