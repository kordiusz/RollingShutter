# Etap 1 (październik 2026): MotionNet, sam ruch kamery

Pierwsze podejście, zakończone 13.10.2026 i porzucone na rzecz RAFT + geometria (etap 2, ruch obiektów).
Kod i notatniki usunięto z repozytorium 11.11.2026; ten plik zostaje jako opis do pracy (rozdz. 5 i 7).
**Ostatni commit z pełnym kodem: `86afe79`** (sposób odtworzenia na końcu).

## Założenie

Scena jest **płaska i nieruchoma**, porusza się tylko kamera (obrót, jego przyspieszenie i przesunięcie w czasie
odczytu). Wtedy całe zniekształcenie RS w obu kamerach opisuje **9 liczb** i dokładny wzór geometryczny.
Wystarczy, że sieć znajdzie te 9 liczb, a resztę liczy geometria. To ten sam pomysł co Velocity-Net
(Zhuang i in., CVPR 2019), rozszerzony na dwie kamery.

## Zbiór `rs_dataset_256`

- Generator `src/synth_rs.py` (zostaje w repo: korzystają z niego `synth_layers.py`, `synth_video.py` i `raft_geom.py`).
- Zdjęcie z DIV2K traktowane jest jako widok GS płaskiej sceny. Losujemy ruch kamery, każdy piksel ma znany czas odczytu
  (wiersz w A, kolumna w B), więc liczymy pozę kamery w tej chwili i pobieramy piksel (jedno `cv2.remap` na obraz).
  Wzorzec i pole przesunięć są dokładne.
- 800 zdjęć DIV2K × 20 próbek, 256×256, ok. **14,4 tys. train / 1,6 tys. val** (podział po zdjęciach).
- Parametry: `configs/config_webcam.json` (zgadywane przed pomiarami kamer): ruch kamery 10–40 px w czasie odczytu.
- Zbiór zajmował ok. 8 GB, bo PNG zaszumionych zdjęć słabo się kompresują. Liczba próbek, nie GB, jest miarą skali.

## Model MotionNet

1. Wspólny (syjamski) ekstraktor cech dla A i B (B obrócony do orientacji A), 1/4 rozdzielczości.
2. **Warstwa korelacji**: podobieństwo cech A z cechami B przesuniętymi o ±6 komórek (±24 px).
3. CNN + warstwy w pełni połączone regresują 9 parametrów ruchu (w pikselach, potem dzielone przez ogniskową).
4. Różniczkowalna geometria RS zamienia parametry na pole przesunięć, obraz A jest przekształcany (`warp`).
- Strata: MSE na polu przesunięć + maskowane L1 na obrazie. 3,6 M wag, Colab T4, 30 epok.

Problemy napotkane w trakcie (warte akapitu w pracy):
- zwykły ReLU w głowicy regresji „umierał” (stała odpowiedź), pomógł LeakyReLU i obcinanie gradientu do 1,0;
- L1 na polu przesunięć zatrzymywało sieć na „zerowym ruchu” (mediana symetrycznych celów), pomogło MSE;
- z obrazu nie da się odczytać kąta widzenia, więc sieć przewiduje ruch w pikselach, a nie w radianach;
- samo sklejenie A i B jako kanałów (bez korelacji) nie uczyło się w krótkich testach; warstwa korelacji to naprawiła.

## Wyniki (val `rs_dataset_256`, wejście bez korekcji 19,66 dB)

| run | model | PSNR [dB] | SSIM | EPE [px] |
|---|---|---|---|---|
| (bez korekcji) | – | 19,66 | 0,527 | – |
| `2026-10-13_motion_dual` | MotionNet, kamery A + B | **28,51** | **0,874** | **0,57** |
| `2026-10-13_motion_single` | MotionNet, tylko kamera A | 19,90 | 0,544 | 3,98 |
| `2026-10-13_flow_dual` | U-Net przewidujący gęste pole, A + B, bez korelacji | 20,45 | 0,588 | 3,15 |

Wnioski:
- **Kamera B jest kluczowa: +8,9 dB z B, +0,2 dB bez niej.** Bez drugiej chwili odczytu nie da się zmierzyć ruchu.
  Ten sam test na ruchu obiektów (X4K) powtarza notatnik `09_ablacje`.
- Model z jawną geometrią i korelacją uczy się, a zwykły U-Net bez korelacji prawie nie (+0,8 dB).
  Ten wniosek potwierdził się później na X4K (zwykły U-Net 28,45 dB wobec 35,81 dB RAFT + geometria + U-Net).
- Pozostały błąd MotionNet to głównie brzegi kadru, które wyszły poza kamerę A. Planowany etap 2 (MotionNet +
  U-Net łączący A i B, notatnik `03_refine_unet`) nigdy nie został uruchomiony.

## Dlaczego porzucony

9 parametrów opisuje tylko ruch **całej** sceny względem kamery. Gdy porusza się obiekt (osoba, samochód, wiatrak),
każdy obszar ma inny ruch i model parametryczny nie ma jak tego wyrazić. Docelowy przypadek pracy to ruch obiektów,
więc od 24.10 pracujemy na zbiorze z nagrań 1000 kl./s (`rs_video_x4k`) z RAFT, który daje osobną prędkość dla
każdego piksela. Ruch samej kamery to szczególny przypadek tego podejścia (zbiór warstwowy `synth_layers.py` zawiera
drżenie kamery).

## Jak odtworzyć kod

```bash
git checkout 86afe79 -- src/model.py src/train.py src/rs_dataset.py src/infer.py notebooks/01_dataset.ipynb notebooks/02_motionnet.ipynb
```

Zawartość: `model.py` (MotionNet, `flow_from_motion`, RefinePipeline), `train.py` (trening i ewaluacja etapu 1),
`rs_dataset.py` (Dataset z polem przesunięć), `infer.py` (inferencja na parach z kamer), notatniki `01_dataset`
(generowanie zbioru) i `02_motionnet` (trening), `notebooks/archiwum/` (pierwszy notatnik z 13.10 i `03_refine_unet`).
Wagi zostały na Google Drive w folderach runów o nazwach z tabeli powyżej (`runs/` albo starsze `MyDrive/rs_runs/`).
