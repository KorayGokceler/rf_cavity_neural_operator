<!-- https://docs.truba.gov.tr/2-temel_bilgiler/interaktif-is-calistirma.html ; modified 2025-07-23 -->

# İnteraktif İş Çalıştırma

Uyarı

ARF hesaplama kümesinde, kullanıcılar derleme ya da benzeri yük getiren işlerini kesinlikle `arf-ui` sunucuları üzerinde gerçekleştirmemelidirler. Bu tür işler için `debug` kuyruğundan `srun` ile interaktif kaynak talep edilmeli ve işlemler hesaplama sunucularında gerçekleştirilmelidir. Ayrıca Open OnDemand üzerinden Linux Desktop alınarak kısa süreli görsel ve interaktif işlerin gerçekleştirilmesi mümkündür. İnteraktif iş çalıştırmak için `debug` kuyruğundan kaynak talep edilmelidir.

`debug` kuyruğu farklı sunucu türlerinden oluştuğundan (hamsi,orfoz,barbun gibi), derleme yapmak için kullanacağınız sunucu türünü -C (constraint) parametresi ile seçebilirsiniz. Sunucu türü belirtilmez ise, sistem otomatik olarak atama yapacaktır. `debug` kuyruğundan `srun` ile kaynak talebi için aşağıdaki komut sekmelerini inceleyebilirsiniz.

debugorfozhamsibarbunbarbun-cudaakya-cuda

```
srun -p debug -N 1 -n 1 -c 110 -A kullanici_adi -J test --time=0:30:00 --pty /usr/bin/bash -i
```

```
srun -p debug -C orfoz -N 1 -n 1 -c 55 -A kullanici_adi -J test --time=0:30:00 --pty /usr/bin/bash -i
```

```
srun -p debug -C hamsi -N 1 -n 1 -c 54 -A kullanici_adi -J test --time=0:30:00 --pty /usr/bin/bash -i
```

```
srun -p debug -C barbun -N 1 -n 1 -c 20 -A kullanici_adi -J test --time=0:30:00 --pty /usr/bin/bash -i
```

```
srun -p debug -C barbun-cuda -N 1 -n 1 -c 20 --gres=gpu:1 -A kullanici_adi -J test --time=0:30:00 --pty /usr/bin/bash -i
```

```
srun -p debug -C akya-cuda -N 1 -n 1 -c 10 --gres=gpu:1 -A kullanici_adi -J test --time=0:30:00 --pty /usr/bin/bash -i
```
