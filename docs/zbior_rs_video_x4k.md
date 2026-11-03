# Zbiór `rs_video_x4k`: pary RS z nagrań 1000 fps

Wersja 1, wygenerowana 2026-10-24 skryptem `src/synth_video.py`.
Zbiór treningowy i walidacyjny dla modelu RAFT + geometria (ruch kamery **i obiektów**).

## 1. Skąd pochodzą dane

**X4K1000FPS** (Sim, Oh, Kim, *XVFI: eXtreme Video Frame Interpolation*, ICCV 2021):
- nagrania 4K (4096×2160) z kamery 1000 fps, sceny plenerowe: piesi, rowerzyści, samochody, ruch ręcznej kamery;
- część treningowa X-TRAIN: **4408 klipów ze 110 scen**, każdy klip to **65 kolejnych klatek (64 ms)** wyciętych jako **768×768 px** z klatki 4K;
- część testowa X-TEST: 15 klipów po 33 klatki w pełnej rozdzielczości 4K;
- licencja: wolno używać do badań i dydaktyki, użycie komercyjne wymaga zgody autorów.

Dlaczego to źródło: przy 1000 fps kolejne klatki dzieli 1 ms, a typowy odczyt kamery RS trwa 10–30 ms. Każdy wiersz
(lub kolumnę) obrazu RS można więc wziąć z klatki nagranej dokładnie w chwili jego odczytu. Wzorzec GS to po prostu
jedna klatka nagrania. Ruch jest prawdziwy (ludzie, pojazdy, kamera z ręki), a nie generowany.

Nazwy plików w X4K (`occ008.320_f2881.mp4`) nie mówią nic pewnego o sile ruchu w klipie.

## 2. Co wzięliśmy

Komenda (uruchomiona lokalnie):
```
python src/synth_video.py --src data/x4k/encoded_train --out data/rs_video_x4k \
    --per-clip 3 --augment --clips-per-scene 9
```
- **110 scen × do 9 klipów** (równomiernie z listy klipów sceny), czyli ok. 990 z 4408 klipów;
- **3 próbki z każdego klipu**, każda z innymi losowymi parametrami i w innej chwili klipu;
- podział train/val **po scenach** (10% scen do val). Ta sama scena nigdy nie jest w obu częściach, więc wynik
  na val nie jest zawyżony podobieństwem do treningu;
- **val: 321 próbek (107 klipów)**; train: _(uzupełnić z `summary.txt`)_;
- rozmiar: ok. 1,2 GB (zip), `MyDrive/rs_project/datasets/rs_video_x4k.zip`.

## 3. Jak powstaje jedna próbka

1. **Losowanie parametrów:** czas odczytu T, przesunięcie startu kamery B (sync), baza między kamerami, wielkość wycinka.
2. **Chwila wzorca t_c:** dokładna klatka nagrania, wybrana tak, żeby cały odczyt obu kamer mieścił się w klipie.
3. **Wycinek:** kwadrat o boku `crop_frac × 768` px w losowym miejscu, zmniejszany do 256×256. Dla kamery B ten sam
   kwadrat przesunięty w prawo o bazę, więc B widzi scenę przesuniętą w lewo, jak kamera stojąca obok.
4. **Czas odczytu każdego piksela** (t = 0 to środek odczytu A):
   - kamera A czyta wiersze: `t_A(y) = (y/255 − 0,5)·T`;
   - kamera B jest obrócona o 90°, więc czyta kolumny: `t_B(x) = sync + ((255 − x)/255 − 0,5)·T`.
5. **Złożenie obrazu RS:** piksel (y, x) bierzemy z klatki nagrania z chwili `t_c + t(y, x)`. Między dwiema
   najbliższymi klatkami (1 ms) interpolujemy liniowo.
6. **Fotometria:** osobne wzmocnienie dla każdej kamery (0,9–1,1, również na jej wzorcu) i szum gaussowski
   σ = 0–0,01 tylko na obrazach RS.
7. **Augmentacja** (`--augment`): losowe odbicie lustrzane klipu i odtwarzanie go wstecz. Oba dają fizycznie poprawne nagranie.
8. **Zapis:** obraz z kamery B jest zapisany obrócony, tak jak oddałaby go prawdziwa kamera B.

## 4. Parametry (`config.json` zbioru)

| parametr | zakres | znaczenie |
|---|---|---|
| `size` | 256 | bok obrazów wyjściowych [px] |
| `readout_ms` | 10–30 | czas odczytu wszystkich wierszy (A) / kolumn (B) wycinka |
| `sync_offset_ms` | −2…+2 | środek odczytu B minus środek odczytu A |
| `baseline_px` | 0–8 | przesunięcie widoku B w px obrazu 256 (jednakowe dla całej sceny) |
| `crop_frac` | 0,5–1,0 | bok wycinka / 768, czyli wycinek 384–768 px zmniejszany do 256 |
| `rot_k` | 1 | obrót kamery B (`np.rot90`) |
| `interp` | linear | mieszanie dwóch najbliższych klatek |
| `gain` | 0,9–1,1 | wzmocnienie kamery |
| `noise_std` | 0–0,01 | szum na obrazach RS |

## 5. Format próbki

Folder `train/<klip>_<nr>/` lub `val/...`:
- `rs_a.png`: obraz RS kamery A (256×256);
- `rs_b.png`: obraz RS kamery B, obrócony jak z kamery;
- `gs_a.png`, `gs_b.png`: wzorce GS w chwili t_c (B obrócony jak `rs_b`);
- `meta.json`: wszystkie parametry, np. `readout_a_s`, `sync_offset_s`, `baseline_px`, `crop` [x0, y0, bok],
  `frame_c`, `source` (klip X4K), `augment`.

Nie ma dokładnego pola przepływu: dla prawdziwego nagrania go nie znamy. Uczymy z błędu obrazu względem wzorca.

## 6. Statystyki zbioru

Liczone skryptem `src/analyze_dataset.py`. **Siła RS próbki** to 95. percentyl przesunięcia pikseli między wzorcem GS
a obrazem RS A (przepływ optyczny Farnebäcka, w px obrazu 256×256). Klasy: słaby < 2 px ≤ średni < 6 px ≤ silny.

_(uzupełnić z `summary.txt`, `hist.png`)_

## 7. Ograniczenia

- **Brak paralaksy zależnej od głębi.** B to przesunięty wycinek tej samej klatki, czyli baza jest jednakowa dla
  całej sceny. W prawdziwym zestawie (kamery ~3 cm od siebie) bliskie obiekty przesuwają się bardziej niż dalekie.
  To najważniejsza różnica względem danych z kamer (widoczna w teście z wiatrakiem).
- **Brak rozmycia ruchu.** Ekspozycja nagrań 1000 fps jest krótsza niż 1 ms, a kamery USB w pokoju naświetlają 10–30 ms.
- **Mały zakres sync (±2 ms).** Niezsynchronizowane kamery USB mogą być przesunięte nawet o pół okresu klatki (~16 ms przy 30 fps).
- **Zmniejszanie wycinka** (1,5–3×) zmniejsza ruch w pikselach o tyle samo.
- **Część próbek prawie bez ruchu** (statyczne fragmenty klipów, nieruchoma kamera). Uczą niewiele i zawyżają PSNR
  wejścia (średnio 26,4 dB na val).
- Interpolacja liniowa między klatkami 1 ms od siebie: przy bardzo szybkim ruchu może dawać lekkie „duchy”.
- Te same parametry obu kamer (T, szum), sceny tylko plenerowe i dzienne.

## 8. Jak wyciągnąć z X4K silniejsze dane: pomiar

Te same 23 klipy z 18 scen, 8 próbek na klip (184 próbki na wariant), bez augmentacji, te same ziarna losowania.
Każdy wariant zmienia jedno ustawienie generatora.

| wariant generatora | p95 przesunięcia, mediana [px] | średnia [px] | słaby | średni | silny | PSNR wejścia [dB] |
|---|---|---|---|---|---|---|
| obecny (wycinek 0,5–1,0, T 10–30 ms) | 4,2 | 7,5 | 28% | 37% | 35% | 26,1 |
| wycinek 1,0 (cały klip 768) | 3,2 | 5,4 | 38% | 38% | 25% | 26,6 |
| wycinek 0,5 (384 px) | 4,8 | 9,3 | 24% | 32% | 45% | 26,0 |
| wycinek 0,334 (256 px, bez skalowania) | 5,4 | 10,7 | 26% | 26% | 48% | 26,5 |
| T = 30 ms stałe | 6,2 | 10,2 | 21% | 28% | 51% | 24,0 |
| przyspieszenie ×2 (co druga klatka) | 7,3 | 11,2 | 15% | 31% | 54% | 23,3 |
| wycinek 0,334–0,5 + T 20–30 ms | 6,9 | 11,6 | 20% | 24% | 55% | 25,2 |
| wycinek 0,334–0,5 + przyspieszenie ×2 | 8,5 | 13,4 | 18% | 20% | 61% | 24,3 |

Wnioski:
- Każda z trzech zmian (mniejszy wycinek, dłuższy odczyt, szybsze odtwarzanie) zwiększa ruch w pikselach.
  Połączone dają około dwa razy silniejszy RS (mediana 4,2 → 8,5 px).
- **15–20% próbek zostaje słabych w każdym wariancie.** To klipy, w których prawie nic się nie rusza.
  Nie pomoże im żadne ustawienie, trzeba je **odrzucać** (pomiar siły RS przy generowaniu albo filtr z `analyze_dataset.py`).
- PSNR wejścia spada mniej, niż rośnie ruch, bo w wielu próbkach rusza się tylko fragment obrazu.
- **Przyspieszenie ×2** (bierzemy co drugą klatkę i traktujemy nagranie jak 1000 fps) jest fizycznie poprawne:
  to ta sama scena, tylko szybsza. Klip ma wtedy 33 klatki (32 ms), więc najdłuższy odczyt to ok. 26 ms,
  a interpolacja odbywa się między klatkami 2 ms od siebie.
- Stałe T = 30 ms daje silny RS, ale model powinien widzieć różne czasy odczytu, więc lepszy jest zakres 15–30 ms.

## 9. Propozycja wersji 2

- wycinek 0,334–0,5, T 15–30 ms, losowo przyspieszenie ×1 lub ×2;
- odrzucanie próbek z p95 przesunięcia < 2 px (ponowne losowanie);
- 20 klipów na scenę × 2 próbki, czyli ok. 4,4 tys. próbek, więcej różnych klipów zamiast wielu próbek z jednego;
- osobno, do badania odporności (nie do głównego treningu): rozmycie ekspozycji (średnia klatek z okna
  ekspozycji), szerszy zakres sync, ciemniejszy obraz i więcej szumu.
