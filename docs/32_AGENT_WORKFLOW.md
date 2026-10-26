# 32 — Ajanlarla çalışma: kurulum, roller, paralel iş, şablonlar

> Bu repo, bir ya da birden çok kodlama ajanıyla (farklı modeller ve araçlar) çalışacak şekilde kuruldu.
> Ortak hafıza repodur. Sohbet geçmişi başka bir modele ya da araca taşınmaz; taşınması gereken her şey
> `AGENTS.md`'ye, `docs/31_SESSION_HANDOFF.md`'ye ve ilgili tasarım belgesine yazılır.

## 1. Repoda ne kurulu

| dosya | kim kullanır | ne işe yarar |
|---|---|---|
| `AGENTS.md` | her ajan, her araç | harita, komutlar, bozulmaması gereken kurallar, çalışma kuralları |
| `docs/31_SESSION_HANDOFF.md` | her ajan | güncel durum, ölçümler, kararlar, açık işler; her oturum sonunda güncellenir |
| `docs/32_AGENT_WORKFLOW.md` | sen ve entegratör | bu rehber |
| `CLAUDE.md` | Claude Code | `AGENTS.md`'yi içe aktarır, Claude'a özel notlar |
| `.claude/settings.json` | Claude Code | izinler: güvenli komutlar sorulmadan çalışır, yıkıcı komutlar engellenir |
| `.claude/agents/*.md` | Claude Code | 8 alt ajan (§3.1): `explorer`, `numerics-check`, `physics-review`, `ui`, `label-pipeline`, `trainer`, `truba-ops`, `docs-writer` |
| `.claude/skills/*/SKILL.md` | Claude Code | 8 komut (§3.2): `/task`, `/verify`, `/handoff`, `/experiment`, `/measure`, `/cst-compare`, `/truba-run`, `/new-family` |
| `scripts/agent_worktree.sh` | herkes | ajan başına ayrı dal + çalışma kopyası (git worktree) |

## 2. Kendi bilgisayarında kurulum

### 2.1 Proje ortamı

```bash
git clone https://github.com/KorayGokceler/rf_cavity_neural_operator
cd rf_cavity_neural_operator
git checkout claude/field-model                # güncel geliştirme dalı
pip install -r requirements.txt -r requirements-web.txt -r requirements-dev.txt
(cd web && npm ci && npm run build)
python -m pytest -q tests/test_field_model.py  # ortam çalışıyor mu
```

### 2.2 Claude Code

1. Claude Code'u kur (resmi yükleyici ya da `npm install -g @anthropic-ai/claude-code`).
2. Repo klasöründe `claude` çalıştır.
3. Giriş yap: abonelikle `/login`, ya da API anahtarı kullanacaksan `ANTHROPIC_API_KEY` ortam değişkeni.
4. Model ve efor seç: `/model` (fable, opus, sonnet, haiku), `/effort`.
5. Kontrol et:
   - `/agents` → sekiz alt ajan görünmeli;
   - `/` yazınca `task`, `verify`, `handoff`, `experiment`, `measure`, `cst-compare`, `truba-run`, `new-family`
     komutları listelenmeli.
6. Kişisel ayarlar `.claude/settings.local.json` dosyasına gider (git'e girmez). `.claude/settings.json`
   herkes için ortaktır; değiştirirken commit et.

Telefondan yönetmek için: aynı klasörde `claude remote-control`. O oturum Claude uygulamasında görünür.

### 2.3 Bulut oturumları (claude.ai/code)

Repo klonlanınca aynı dosyalar otomatik yüklenir. Fark: bulut konteyneri geçicidir, commit edilip push
edilmeyen her şey kaybolur. Oturum sonunda `/handoff` ve push.

### 2.4 Diğer araçlar ve modeller

`AGENTS.md`'yi okuyan araçlar talimatları kendiliğinden alır. Okumayanlara ilk mesajda
"AGENTS.md ve docs/31_SESSION_HANDOFF.md'yi oku" de. Claude'a özel dosyalar (`.claude/`) başka
araçlarda bir şey yapmaz; kurallar zaten `AGENTS.md`'de.

## 3. Roller ve model seçimi

| rol | ne yapar | önerilen |
|---|---|---|
| **Entegratör** | işleri böler, dalları birleştirir, `docs/31`'i tutar | en güçlü model; Claude'da Opus 5.5 ya da Fable 5.1, yüksek efor |
| **Uygulayıcı** | tek bir görevi kendi dalında yapar | güçlü model, orta efor; rutin arayüz ve doküman işinde Sonnet 5.5 |
| **İnceleyici** | diff'i temiz bağlamla inceler | uygulayıcıdan **farklı** bir model ya da sağlayıcı; Claude içinde `physics-review` |
| **Doğrulayıcı** | testleri ve analitik kontrolü çalıştırır, sayıları raporlar | `numerics-check` / `ui` alt ajanları ya da `/verify` |
| **Kaşif** | kod arama, uzun log özetleme | ucuz model; Claude'da `explorer` (Haiku 5.5) |

### 3.1 Alt ajanlar (`.claude/agents/`)

| ajan | ne zaman | yazar mı | model |
|---|---|---|---|
| `explorer` | kod arama, bir değerin hat boyunca izlenmesi, uzun log / belge özeti | hayır | ucuz |
| `numerics-check` | çözücü, etiket, operatör, model, kayıp, QoI değişikliğinden sonra testler + analitik pillbox + Ritz sınırı kontrolleri; yalnız rapor | hayır | orta |
| `physics-review` | commit'ten önce diff'in fizik / sayısal yöntem incelemesi (değişmezler) | hayır | en güçlü |
| `ui` | web arayüzü: değişiklik, derleme, Playwright tarayıcı kontrolü, ekran görüntüleri, süreler | evet | orta |
| `label-pipeline` | geometri ailesi → eğri mesh → p3 etiket → H5; hatalı / yavaş örnekler, STEP; üreteç davranışı sahibin onayıyla | evet | orta |
| `trainer` | eğitim ve değerlendirme, kontrollü deneyler, taban çizgileri (GP, kaba FE) | evet | orta |
| `truba-ops` | TRUBA betikleri, kaynak planı, `--dry-run`, log okuma; iş göndermez | evet | orta |
| `docs-writer` | numaralı belgeler, docs/31, makale (LaTeX); her sayının kaynağı | evet | orta |

### 3.2 Komutlar (`.claude/skills/`)

| komut | ne yapar |
|---|---|
| `/task` | işi başlatır: oku, yeniden ifade et, dal, plan + kabul ölçütleri, onay bekle |
| `/verify` | değişen yollara göre kontrolleri seçer ve sayıları raporlar |
| `/handoff` | oturum sonunda docs/31'i günceller |
| `/experiment` | kontrollü eğitim deneyi: soru ve kollar önce, kimliğe göre bölme, tohumlar, taban çizgileri, docs/34–35'e kayıt |
| `/measure` | önce / sonra süre ve bellek: sabit girdi, ısınma, ≥ 5 koşu, iş parçacığı denetimi, tablo |
| `/cst-compare` | CST kıyası: kalibrasyon seti, CST ayarları, kural farkları (R/Q, L_acc), tablolar |
| `/truba-run` | TRUBA koşusu hazırlar: kaynaklar ölçümden, `--dry-run`, sahibe komutlar, log takibi |
| `/new-family` | yeni geometri ailesi: kurucu, sabit kimlik bloğu, testler, belgeler (sahibin onayıyla) |

İlkeler:
- **Farklı modeller farklı hataları yakalar.** Çok modelli kurulumun asıl getirisi çapraz inceleme.
- **Hakem testlerdir.** Bir modelin başka bir modelin işine "doğru" demesi kanıt değildir.
- **Maliyet:** en güçlü modeli plan, inceleme ve zor matematik için yüksek eforda kullan. Uygulamayı
  aynı modelle daha düşük eforda çalıştırmak çoğu zaman yeterli. Ana oturumda model değiştirme; ucuz
  modelleri alt ajanda kullan, önbellek korunur.

## 4. Paralel çalışma

Aynı dosyalarda iki ajan aynı anda çalışmaz. Her ajana ayrı dal ve ayrı çalışma kopyası:

```bash
bash scripts/agent_worktree.sh data-cache          # ../rf_cavity_neural_operator-data-cache, dal agent/data-cache
bash scripts/agent_worktree.sh ui-field-model
bash scripts/agent_worktree.sh --list
bash scripts/agent_worktree.sh --remove data-cache # değişiklik varsa reddeder; dal kalır
```

- Paralel iş için **birbirine dokunmayan** konular seç (ör. veri hattı + arayüz + doküman).
  Aynı modüle dokunan işleri sıraya koy.
- Arayüz işi yapan worktree'de `cd web && npm ci` gerekli; `node_modules` paylaşılmıyor.
- Veri dosyaları (`*.h5`, `*.pkl`) git'te değil; her worktree'ye ayrıca göster ya da paylaşılan bir klasöre işaret et.
- Birleştirme: uygulayıcı dalını push eder; entegratör doğrulamadan sonra geliştirme dalına birleştirir.
  Başkasının dalında geçmiş yeniden yazılmaz (rebase ya da force-push yok).

## 5. Bir görevin yaşam döngüsü

```
1. seç        docs/31 §6'dan bir iş (sahip ve dal sütununu doldur)
2. hazırla    dal / worktree; /task: oku, ölç, plan + kabul ölçütleri → onay bekle
3. uygula     küçük, odaklı değişiklikler
4. doğrula    /verify: testler, analitik kontrol, build + tarayıcı; sayılar
5. incele     başka bir model ya da physics-review; bulgular düzeltilir
6. birleştir  entegratör; commit mesajında sayılar
7. devret     /handoff: docs/31 güncel, push
```

## 6. Mesaj şablonları

Her araçta ve modelde aynı şekilde yapıştırılabilir.

**Uygulayıcı (görev başlatma):**
```
Önce AGENTS.md ve docs/31_SESSION_HANDOFF.md'yi oku.
Görev: <tek cümle>  (docs/31 §6 madde <n>)
Dal: <agent/...>. Dokunma: <dosyalar / davranışlar>.
Bitti sayılması için: <testler>, <sayı eşiği>, <ekran görüntüsü / süre>.
Önce mevcut durumu ölç ve planını yaz; onaylarsam uygula. Sonunda docs/31'i güncelle.
```

**İnceleyici:**
```
AGENTS.md'deki kurallara göre şu diff'i incele: git diff <taban>...<dal>.
Fizik ve sayısal yöntem hatalarına odaklan (projeksiyon, Ritz üst sınırı, birimler, float64,
torch/ngsolve sırası, eğri mesh kuralları). Her bulgu için dosya:satır, somut hata senaryosu, öneri.
Bulgu yoksa "bulgu yok" de; üslup yorumu ekleme.
```

**Doğrulayıcı:**
```
AGENTS.md'yi oku. <dal> dalındaki değişiklik için /verify adımlarını uygula
(ya da: değişen dosyalara göre testleri seç ve çalıştır). Komutları, geçen/kalan sayılarını ve ana
sayıları raporla. Kod düzeltme.
```

**Entegratör (oturum başı):**
```
AGENTS.md ve docs/31'i oku. Açık işlerden birbirine dokunmayan 2–3 tanesini seç; her biri için
dal adı, kabul ölçütü ve uygun rolü (model) öner. Onaydan sonra docs/31 §6'ya sahip ve dal yaz.
```

**Oturum sonu:**
```
/handoff  (ya da: docs/31'in §1, §3, §4, §6 bölümlerini bu oturuma göre güncelle, yalnız dokümanı commit et)
```

## 7. İyi prompt alışkanlıkları

- **Hedef + kısıt + kabul ölçütü** her mesajda. "Hızlandır" değil: "adım başına süreyi ölç, %30
  düşür, pillbox hatası 1e-5'in altında kalsın".
- **Önce plan,** büyük değişikliklerde onayla sonra uygulat.
- **Log ve sayıları olduğu gibi yapıştır.**
- **Konu değişince yeni oturum aç.** Bağlam kısa kalır, maliyet düşer.
- **Aşırı kural yazma.** Yeni modeller talimatları harfiyen uyguluyor; kısa ve net daha iyi sonuç veriyor.
- **Anlamadığın yerde açıklama iste.** Fizik ya da nümerik kodda bir değişikliği anlamadan birleştirme.

## 8. Sık karşılaşılan tuzaklar

| durum | çözüm |
|---|---|
| Segfault (torch + ngsolve) | süreçte önce `import torch`, sonra `ngsolve` |
| Kaydedilen alan DOF sayısı tutmuyor | eğri mesh'i `highorder.single_level` ile kaydet (docs/30 §5) |
| Playwright tarayıcı indirmeye çalışıyor | bulut konteynerinde Chromium hazır: `executable_path` ile `/opt/pw-browsers` altındakini kullan |
| UI değişikliği görünmüyor | `npm run build` sonrası sunucuyu yeniden başlat; `serve_web.py` `dist`'i kaynak değişince yeniden kurar |
| TRUBA komutu | her zaman önce `--dry-run` |
| Bulut oturumu bitti, iş kayboldu | commit + push olmadan oturumu kapatma; `/handoff` |
