# Analiza zbioru `rs_video_x4k` i propozycje poprawek (agent, 2026-11-03)

Analiza lokalna, bez treningu sieci. Wszystkie pliki pomocnicze są w `data/analiza_agent/`
(skrypty w `data/analiza_agent/skrypty/`), zbiór i źródło nie były zmieniane.

## 0. Najkrócej

- **RS jest widoczny, ale tylko w części zbioru.** W ok. 40% próbek (klasa „silny”, ruch kamery, szybcy ludzie)
  pochylenia i rozciągnięcia widać gołym okiem. W 33% próbek (p95 < 2 px) obraz RS praktycznie nie różni się od GS,
  a różnica A–B to wtedy głównie przesunięcie bazy.
- **Generator v1 jest poprawny** (odtworzenie 200 próbek z klipów źródłowych: 45 dB, wybór klatki `frame_c`
  jednoznaczny, kierunek odczytu B zgodny z `b_rot_k = 1`). Dwie wady: w meta brakuje flag augmentacji
  (odbicie / wstecz), a przy szybkim ruchu (16% próbek ma > 2 px ruchu na 1 ms) interpolacja liniowa
  daje „duchy” i sztuczne rozmycie.
- **Siłę RS wyznacza klip, nie parametry generatora**: klip wyjaśnia 89% wariancji log(p95), scena 73%,
  czas odczytu i skala wycinka tylko trochę (korelacja rang +0,25 i −0,12).
- **Różnorodność daje scena, a nie kolejne próbki z klipu**: dwie próbki z tego samego klipu są do siebie
  podobne jak 0,76 (przecięcie histogramów), z różnych klipów tej samej sceny 0,49, z różnych scen 0,17.
- **Zbiór jest daleko od wiatraka**: brak obrotu, brak rozmycia, prędkości 0,4 px/ms (mediana) wobec 3,5–7 px/ms
  końca łopatki przy 300–600 obr/min. Trzeba go uzupełnić proceduralnym generatorem obracających się obiektów
  (szkic działa: `skrypty/demo_wiatrak.py`).

## 1. Co sprawdzono i czym

| krok | skrypt | wynik |
|---|---|---|
| arkusze porównawcze (8 próbek na arkusz) | `skrypty/arkusze.py` | `arkusze/*.jpg` |
| odtworzenie 200 losowych próbek z klipów X4K | `skrypty/odtworz.py --n 200` | `odtworzenie.csv` |
| rozkłady siły RS, zależności, wykorzystanie źródła | `skrypty/rozklady.py` | `rozklady.txt` |
| podobieństwo treści: klip / scena / różne sceny | `skrypty/roznorodnosc.py` | `roznorodnosc.txt` |
| warianty generatora v2 (40 klipów z 40 scen × 4 próbki) | `skrypty/warianty.py` | `warianty/warianty*.csv`, `.txt` |
| szkic generatora wiatraka | `skrypty/demo_wiatrak.py` | `arkusze/demo_wiatrak.jpg` |

Miara siły RS jak w `src/analyze_dataset.py`: 95. percentyl długości przepływu Farnebäcka GS A → RS A
w pikselach obrazu 256×256 (klasy: słaby < 2 ≤ średni < 6 ≤ silny).

## 2. Czy efekt RS jest widoczny (arkusze)

Kolumny każdego arkusza: **RS A | RS B obrócony do orientacji A (`np.rot90(rs_b, -1)`) | GS A |
|RS A − GS A|×4 | nałożenie RS A (czerwony) i RS B (turkus)**. Pod obrazami: T, sync, baza, skala wycinka,
PSNR wejścia, p95 przesunięcia, klasa i typ ruchu.

Arkusze: [losowe train](../data/analiza_agent/arkusze/losowe_train.jpg),
[losowe val](../data/analiza_agent/arkusze/losowe_val.jpg),
[klasa słaby](../data/analiza_agent/arkusze/klasa_slaby.jpg),
[klasa średni](../data/analiza_agent/arkusze/klasa_sredni.jpg),
[klasa silny](../data/analiza_agent/arkusze/klasa_silny.jpg),
[ruch globalny](../data/analiza_agent/arkusze/ruch_globalny.jpg),
[sceny 1](../data/analiza_agent/arkusze/sceny_1.jpg), [sceny 2](../data/analiza_agent/arkusze/sceny_2.jpg),
[sceny val](../data/analiza_agent/arkusze/sceny_val.jpg),
powiększenie dwóch silnych próbek: [zoom_00788_00725.png](../data/analiza_agent/arkusze/zoom_00788_00725.png).

Co widać:
- **Silny RS (p95 ≥ 6 px, 38% próbek) jest wyraźny.** Najlepszy przykład: `train/00788_02` (scena 138, p95 36 px,
  ruch kamery w pionie). W GS poręcz jest pozioma, w RS B (czyta kolumny) jest **pochylona o ok. 10°**,
  a w RS A (czyta wiersze) pozostaje pozioma, ale cały obraz jest **rozciągnięty w pionie** (twarz i poręcz niżej niż w GS).
  Przy ruchu poziomym (np. `train/00098_02`, `00725_00`) jest odwrotnie: w A pochylają się pionowe słupki,
  a B jest rozciągnięty lub ściśnięty w poziomie.
- **A i B nie są „wygięte w przeciwne strony”, tylko w prostopadłe.** Kamera A pochyla to, co rusza się
  w poziomie, i rozciąga to, co rusza się w pionie; kamera B na odwrót. Przesunięcie względem GS zależy w A od wiersza,
  w B od kolumny, i jest takie samo na antyprzekątnej (y = 255 − x przy sync = 0). Na nałożeniach A/B ruchome obiekty
  mają czerwoną i turkusową kopię rozsuniętą w różnych kierunkach (np. idący mężczyzna w `train/00341_02`).
  Przeciwne wygięcia pojawiają się dopiero przy obrocie (patrz demo wiatraka w p. 7).
- **Na mapach błędu przy ruchu globalnym widać ciemny poziomy pas w połowie wysokości**: to wiersz czytany
  dokładnie w chwili GS (t = 0). Potwierdza konwencję czasu (t = 0 w środku odczytu A).
- **Klasa „słaby” (33%) wygląda statycznie**: RS A i GS są nie do odróżnienia, błąd tylko na krawędziach (< 2 px).
  Różnica między A i B (nałożenie) to wtedy prawie wyłącznie przesunięcie o bazę (0–8 px), nie RS.
  Na 8 losowych próbkach train 3 wyglądały na statyczne, na val 1–2 (zgodne z 33% / 36% z `summary.txt`).
- **Klasa „średni” (2–6 px)**: bez obrazu GS obok RS zwykle **nie da się zauważyć gołym okiem**; zniekształcenie
  dotyczy małego fragmentu (ręka, noga, koło roweru).
- **Miara p95 zaniża ruch cienkich struktur**: rowerzyści w `val/00136_02` mają p95 = 1,6 px („słaby”), choć szprychy
  i nogi wyraźnie się ruszają. Filtr `--min-disp` może więc wyrzucać część próbek z obrotem kół.
- W 2 z 8 próbek arkusza „ruch globalny” powtarzają się te same sceny (014, 129): kilka scen z ruchem kamery
  dominuje klasę silną (p. 4).

## 3. Poprawność generatora (odtworzenie z klipu źródłowego)

200 losowych próbek (train + val). Dla każdej: wczytany klip X4K, szukanie augmentacji (4 kombinacje odbicia i odwrócenia,
bo meta ich nie zapisuje), wycinek `crop`, klatka `frame_c`, dopasowane wzmocnienie; potem złożenie RS A i RS B
wzorami v1 i porównanie z plikami. Wyniki (`odtworzenie.csv`):

| sprawdzenie | wynik (min / mediana / max) | wniosek |
|---|---|---|
| PSNR `gs_a` vs klatka `frame_c` wycinka | 39,6 / 58,9 / 120 dB | to dokładnie ta klatka |
| PSNR `gs_a` vs klatka sąsiednia (k ± 1) | 17,5 / 37,1 / 50,4 dB | `frame_c` wyróżniona; najmniejszy zapas 3,4 dB (scena statyczna) |
| PSNR odtworzonego `rs_a` | 32,7 / 45,4 / 59,4 dB | zgodne (szum σ ≤ 0,01 daje ≥ 40 dB) |
| PSNR odtworzonego `rs_b` (`b_rot_k = 1`, z sync) | 30,7 / 45,1 / 59,6 dB | zgodne |
| to samo przy odwrotnym kierunku odczytu kolumn (k = 3) | 11,7 / 22,8 / 44,0 dB | gorzej w 199/200: kierunek z meta jest właściwy |
| to samo bez sync | 18,5 / 37,6 / 55,9 dB | gorzej w 200/200: sync jest zastosowany |
| wzmocnienie dopasowane / `gain_a` z meta | 0,96–1,00 | zgodne (różnica przez przycięcie do [0, 1]) |

- Jedyny „odstający” przypadek (`train/00583_02`, 32,7 dB) to `frame_c = 32`, czyli środek klipu: klatka GS jest ta sama
  przy odtwarzaniu wprost i wstecz, a mój skrypt wybrał zły kierunek. Po odwróceniu: 40,3 dB. Generator działa dobrze.
- **Konwencja czasu** zgodna z opisem: `t_A(y) = (y/255 − 0,5)·T`, `t_B(x) = sync + ((255 − x)/255 − 0,5)·T`
  (B czyta kolumny od prawej). Zapis `rs_b = rot90(·, 1)`: wiersz 0 zapisanego obrazu B to kolumna 255 w kadrze A,
  czytana pierwsza, więc obraz zapisany jest spójny z czasem odczytu. **Nie da się tu sprawdzić, czy to zgadza się
  z prawdziwym zestawem** (w którą stronę obrócona jest kamera B); to trzeba potwierdzić na nagraniu wiatraka
  (zła orientacja = łopatki w B wygięte w złą stronę).
- **Baza**: B to ten sam wycinek przesunięty w prawo o `baseline_src_px` (np. 2 px źródła przy bazie 0,57 px w 256).
  Działa, ale jest jednakowa dla całej sceny (znane ograniczenie).
- **Brak flag augmentacji w meta** (`augment: true` nie mówi, czy było odbicie / odwrócenie). Augmentacja trafia
  do próbek prawie równo: odbicie 101/200, wstecz 102/200. `frame_c` odnosi się do klipu po odwróceniu,
  więc bez szukania nie da się wrócić do klatki źródła.
- **„Duchy” interpolacji liniowej**: ruch między kolejnymi klatkami (1 ms) w chwili t_c, p95 w px obrazu 256:
  mediana 0,49 px, **> 1 px w 31% próbek, > 2 px w 16%**, maksimum ~10 px. Powyżej ~2 px/ms mieszanie dwóch klatek
  daje podwójne krawędzie i rozmycie, którego nie ma w GS; metoda „najbliższa klatka” daje zamiast tego schodki
  co 256/T wierszy (ok. 13 wierszy przy T = 20 ms). Przykład: [duchy_interpolacji.png](../data/analiza_agent/arkusze/duchy_interpolacji.png)
  (kolumny: RS ze zbioru | odtworzony liniowo | najbliższa klatka | GS; górny wiersz `train/00017_02` ~10 px/ms,
  dolny `train/00036_02` ~6 px/ms: siatka ogrodzenia w RS jest rozmyta, krawędzie pudełek podwojone).
  PSNR „liniowo vs najbliższa klatka”: 5. percentyl 33 dB, mediana 48 dB, czyli dotyczy to kilkunastu procent próbek,
  a przyspieszenie ×2 / ×3 zwiększa krok 2–3 razy (p. 6).
- Drobne: mp4 z X4K mają okresowy (co 4 klatki) wzór różnic między klatkami nawet w statycznej scenie
  (średnia |różnica| 0,83–1,35 poziomu szarości, scena 105). To szum kompresji, który w obrazie RS układa się w poziome pasy;
  przy sile RS < 1 px to on dominuje błąd wejścia.

## 4. Rozkład siły RS (z `data/analiza_rs_video_x4k~/samples.csv`)

| | train | val |
|---|---|---|
| próbek / klipów / scen | 2397 / 799 / 97 | 321 / 107 / 13 |
| p95 przesunięcia, kwantyle 5/25/50/75/95% [px] | 0,57 / 1,53 / 3,86 / 10,1 / 26,1 | 0,57 / 1,37 / 2,99 / 7,85 / 24,6 |
| słaby / średni / silny | 32,8 / 28,3 / 38,9% | 35,8 / 32,4 / 31,8% |
| p95 < 1 px („prawie statyczne”) | 14,2% | 15,9% |
| ruszający się fragment < 10% pikseli | 22,2% | 26,5% |
| ruch brak / lokalny / globalny | 32,8 / 58,7 / 8,6% | 35,8 / 53,6 / 10,6% |
| PSNR wejścia: słaby / średni / silny [dB] | 33,7 / 26,3 / 19,0 | 32,8 / 25,8 / 19,7 |
| p95 mediana: ruch lokalny / globalny [px] | 6,9 / 12,1 | 5,2 / 15,1 |

Zależności (wszystkie próbki, `rozklady.txt`):

| czas odczytu T [ms] | 10–15 | 15–20 | 20–25 | 25–30 |
|---|---|---|---|---|
| p95 mediana [px] | 2,4 | 3,2 | 4,1 | 6,1 |
| słaby / silny | 45 / 27% | 37 / 36% | 29 / 40% | 20 / 51% |

| skala wycinka (px źródła / 256) | 1,5–1,9 | 1,9–2,3 | 2,3–2,7 | 2,7–3,0 |
|---|---|---|---|---|
| p95 mediana [px] | 4,4 | 4,0 | 3,6 | 2,9 |
| słaby / silny | 28 / 41% | 28 / 39% | 36 / 38% | 43 / 33% |

- Korelacja rang (Spearman) p95 z: T **+0,25**, skalą **−0,12**, |sync| 0,00. Parametry generatora tylko przesuwają
  rozkład; o sile RS decyduje treść klipu.
- **Udział wariancji log(p95) wyjaśniony przez klip: 89%, przez scenę: 73%.** Trzy próbki z jednego klipu mają
  podobną siłę (stosunek max/min w klipie: mediana 1,95).
- **22,3% klipów ma wszystkie 3 próbki słabe** (< 2 px), 35% wszystkie < 3 px. Tych klipów nie poprawi żaden parametr.
- Sceny: mediana p95 sceny 0,9 / 1,6 / **4,3** / 10,0 / 20,6 px (kwantyle 5/25/50/75/95%). **34 ze 110 scen ma medianę
  < 2 px**, w 27 scenach ≥ 2/3 próbek jest słabych (najsłabsze: 093, 037, 105, 085, 123, 164, 029, 095, 126, 149).
  Odwrotnie: 10 najsilniejszych scen daje 26% wszystkich silnych próbek, 20 scen 49%.
- Val jest trochę słabszy od train (mediana 2,99 vs 3,86 px, silnych 32% vs 39%) przy tylko 13 scenach: wynik na val
  zależy mocno od tych kilku scen.
- Ile próbek „niewiele uczy”: 33% ma p95 < 2 px (dla nich prawie każde wyjście ≈ wejście jest dobre), kolejne ok. 11%
  ma 2–3 px. Razem **ok. 44% zbioru uczy głównie odwzorowania tożsamościowego** i zawyża PSNR wejścia (26,4 dB).

## 5. Wykorzystanie źródła X4K

| | wartość |
|---|---|
| scen z klipami / klipów w X-TRAIN | 110 (folder 111 jest pusty) / 4408 |
| klipów na scenę (kwantyle 5/25/50/75/95%) | 2 / 18 / 45 / 60 / 73 |
| **klipów użytych przez v1** | **906 (20,6%)**, po 3 próbki = 2718 próbek |
| klatek na próbkę | T + 2·\|sync\| + 2 ≈ 12–36 z 65; 3 próbki z klipu w losowych t_c nakładają się czasowo |
| odstęp startów klipów w scenie (`_fNNNN`) | mediana 80 klatek; **44% sąsiednich klipów nakłada się w czasie** (< 65 klatek) |
| korelacja liczby `occXXX.YYY` z nazwy z p95 | Spearman +0,14 (w scenie +0,18): nazwa nie nadaje się do wyboru klipów |

Podobieństwo treści obrazów GS (`roznorodnosc.txt`, przecięcie histogramów HSV, 0 = różne, 1 = identyczne;
900 losowych próbek):

| para próbek | podobieństwo (mediana) | \|różnica log p95\| (mediana) |
|---|---|---|
| ten sam klip | **0,76** | 0,40 |
| ta sama scena, inny klip | 0,49 | 0,51 |
| różne sceny | 0,17 | 1,17 |

Wnioski:
- **Więcej próbek z jednego klipu: nie** (najwyżej 2). Kolejna próbka z klipu to prawie ta sama treść i ta sama siła ruchu;
  zmienia się tylko wycinek i T, a to v1 i tak losuje.
- **Więcej klipów na scenę: tak, umiarkowanie** (z 9 do ~20–25). Klip z innej chwili tej samej sceny daje inną
  treść (0,49) i inną siłę ruchu. Powyżej ~25 klipów rośnie nakładanie się w czasie (44% sąsiednich klipów się pokrywa).
- **Główny ogranicznik to 110 scen.** Ok. 1/3 z nich jest prawie statyczna. Więcej realnej różnorodności dałyby
  nowe źródła (inne nagrania wysokiej szybkości, własne nagrania, generator proceduralny z p. 7), nie mnożenie próbek z X4K.

## 6. Warianty generatora (próba na 40 klipach)

`skrypty/warianty.py`: 40 losowych klipów z 40 różnych scen, 4 próbki z klipu, bez augmentacji, **te same ziarna
losowania dla każdego wariantu** (`make_sample` z `src/synth_video_v2.py`). Obrazy nie były zapisywane, tylko liczby
(`warianty/warianty.csv`). Kolumna „krok klatki” to p95 ruchu między dwiema kolejnymi klatkami (po przyspieszeniu)
w chwili t_c, w px obrazu 256: powyżej ~2 px interpolacja liniowa daje widoczne „duchy” (p. 3).
`min-disp 2` = ponowne losowanie jak `--min-disp 2 --max-tries 4`.

Uwaga: te 40 klipów jest silniejszych niż przeciętny klip v1 (wariant v1 ma tu medianę 7,0 px, w całym zbiorze 3,9 px),
bo v1 wybiera klipy równomiernie po nazwie, a tu wybór był losowy. Porównywać należy warianty między sobą.

| wariant | próbek | p95 mediana [px] | średnia | słaby | średni | silny | PSNR wej. [dB] | krok > 1 px | **krok > 2 px** | losowań / próbkę |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 (wycinek 0,5–1, T 10–30) | 160/160 | 7,0 | 9,4 | 24% | 19% | 57% | 23,4 | 49% | **19%** | 1,00 |
| wycinek 256–384 px | 160/160 | 11,1 | 13,6 | 22% | 16% | 62% | 24,1 | 64% | **37%** | 1,00 |
| T 15–30 | 160/160 | 8,3 | 10,6 | 20% | 19% | 61% | 22,7 | 49% | **19%** | 1,00 |
| przyspieszenie ×2 | 160/160 | 14,4 | 16,3 | 11% | 17% | 72% | 20,6 | 72% | **52%** | 1,00 |
| przyspieszenie ×3 (T 10–18) | 143/160 | 16,0 | 17,0 | 9% | 20% | 71% | 20,4 | 80% | **67%** | 1,00 |
| v2 z dokumentu: 256–384, T 15–30, ×1/×2 | 160/160 | 15,8 | 17,4 | 13% | 18% | 69% | 22,2 | 69% | **49%** | 1,00 |
| v2 z dokumentu + min-disp 2 | 155/160 | 16,9 | 19,3 | 0% | 24% | 76% | 20,6 | 78% | **56%** | 1,24 |
| v2 + ×1/×2/×3, T 12–30 + min-disp 2 | 157/160 | 16,1 | 19,9 | 0% | 21% | 79% | 20,6 | 82% | **61%** | 1,34 |
| v1 + min-disp 2 | 140/160 | 8,9 | 11,3 | 0% | 33% | 67% | 21,2 | 58% | **22%** | 1,52 |
| 256–384, T 15–30, ×1 + min-disp 2 | 147/160 | 15,1 | 16,6 | 0% | 28% | 72% | 21,4 | 72% | **40%** | 1,43 |
| **256–512, T 15–30, sync ±4, ×1 + min-disp 2 (proponowany)** | 144/160 | 13,3 | 15,6 | 0% | 26% | 74% | 21,1 | 73% | **34%** | 1,45 |
| *szacunek:* v1, ×2 tylko gdy krok przy ×2 ≤ 2 px (adaptacyjnie) | 160/160 | 9,4 | 11,0 | 14% | 19% | 67% | 21,8 | – | **19%** | 1,00 |

Ostatni wiersz to szacunek z par próbek o tym samym ziarnie (v1 i ×2 losują ten sam wycinek, różnią się T i t_c):
dla każdej próbki biorę wersję ×2, jeśli jej krok klatki ≤ 2 px, w przeciwnym razie ×1. ×2 trafia wtedy do 47% próbek.

Wnioski:
- **Każda zmiana wzmacnia RS, zgodnie z p. 8 opisu zbioru**: wycinek 256–384 px +59% mediany, ×2 +106%,
  razem (v2 z dokumentu) +126%. Dłuższy odczyt (T 15–30) daje mniej (+19%).
- **Ale przyspieszenie i mały wycinek zwiększają ruch na krok klatki tak samo jak RS**, więc rośnie udział próbek
  z „duchami” interpolacji: z 19% (v1) do 49–56% (v2 z dokumentu), przy ×2 w samych próbkach ×2 aż 66–72%.
  W wersji 2 z dokumentu ponad połowa zbioru miałaby sztuczne podwójne krawędzie, których nie ma w GS.
- **×3 nie ma sensu**: klip ma wtedy 22 klatki (21 ms), więc T trzeba ograniczyć do ≤ 18 ms; 11% losowań się nie mieści,
  a krok > 2 px ma 67% próbek.
- **`--min-disp 2` działa i jest tani**: usuwa wszystkie słabe próbki, średnio 1,24–1,52 losowania na próbkę.
  Przy v1 aż 20/160 próbek (12,5%) nie przechodzi po 4 losowaniach (statyczne klipy), przy v2 tylko 3–5/160.
- **Przyspieszenie adaptacyjne** (×2 tylko dla wolnych klipów) daje połowę zysku ×2 (słabych 24% → 14%,
  mediana 7,0 → 9,4 px) bez dodatkowych „duchów”. To wymaga małej zmiany w `make_sample` (p. 8).
- **Bez przyspieszenia, z małym wycinkiem, T 15–30 i filtrem** (dwa przedostatnie wiersze tabeli) RS jest prawie tak silny
  jak w v2 z dokumentu (mediana 13,3–15,1 vs 16,9 px, 0% słabych, 72–74% silnych), a duchów jest mniej (34–40% vs 56%).
  Część duchów jest nieunikniona przy interpolacji liniowej: filtr zostawia właśnie szybkie klipy (v1 + filtr: 22%).
  Wycinek 256–512 zamiast 256–384 daje trochę słabszy RS, ale mniej duchów i więcej kontekstu w kadrze.

## 7. Różnice względem docelowego wiatraka

| cecha | zbiór `rs_video_x4k` | wiatrak w pokoju |
|---|---|---|
| typ ruchu | postępowy (ludzie, auta, kamera z ręki); obrót tylko w kołach rowerów, mały | obrót wokół osi, kierunek prędkości zmienia się w kadrze |
| prędkość [px/ms, obraz 256] | mediana 0,39, p95 2,4, p99 3,7; > 3,5 px/ms tylko 1,4% próbek, > 6,9 tylko 0,1% | koniec łopatki r = 110 px: 3,5 px/ms przy 300 obr/min, 6,9 przy 600, 13,8 przy 1200 |
| ruch w czasie odczytu | prawie stała prędkość | obrót o ωT: 36° przy 300 obr/min i T = 20 ms, 72° przy 600 |
| rozmycie ruchu | brak (ekspozycja < 1 ms) | ekspozycja 10–30 ms w pokoju: smugi na łopatkach |
| sync A–B | ±2 ms | kamery USB bez synchronizacji: do ±16 ms przy 30 fps |
| paralaksa | jednakowa baza dla całej sceny | wiatrak blisko kamer, tło daleko |

Ważne dla modelu RAFT + geometria: geometria zakłada **stałą prędkość piksela w czasie odczytu** (`p = q + u·t`).
Przy obrocie o 72° w czasie odczytu łuk różni się od cięciwy o ok. r·θ²/2 ≈ 0,2·r, czyli **ok. 20 px na końcu łopatki**
(r = 110 px, 600 obr/min, T = 20 ms). Tego zbiór X4K w ogóle nie pokazuje.

Czym uzupełnić:
1. **Proceduralny generator obracających się obiektów** (szkic: `skrypty/demo_wiatrak.py`,
   [demo_wiatrak.jpg](../data/analiza_agent/arkusze/demo_wiatrak.jpg)). Łopatki (maska + tekstura z dowolnego obrazu)
   obracają się nad tłem z X4K / DIV2K; piksel (y, x) w chwili t pobierany jest z układu wirnika obróconego o −ω·t,
   czyli jedno `cv2.remap` na kamerę. Zalety: ciągły czas (zero „duchów”), dowolne ω, **dokładny przepływ jako
   etykieta** (można uczyć RAFT nadzorowanie), dowolny sync i rozmycie (średnia z okna ekspozycji). W demo łopatki
   są w A i B wygięte wyraźnie inaczej, a przy ekspozycji 8 ms i 600 obr/min powstają smugi jak z kamery USB.
   Parametry do losowania: 2–6 łopatek, ω 100–1500 obr/min, oba kierunki, środek w dowolnym miejscu kadru,
   promień 40–150 px, ruch tła (kamera) 0–1 px/ms, ekspozycja 0–T.
2. **Rozmycie ekspozycji w X4K** (średnia z klatek z okna [t − E, t] dla każdego wiersza; E co 1 ms, więc koszt
   E złożeń). Jako osobny wariant odporności, ok. 20–30% próbek.
3. **Szerszy sync** (±4 ms w głównym zbiorze; ±8–16 ms tylko z ×1 i krótszym T, bo klip ma 64 ms).
4. **Nagrania własne / inne zbiory wysokiej szybkości** z obrotem (wentylatory, koła) jako walidacja, a nie trening.

## 8. Lista poprawek (priorytet)

| prio | poprawka | decyzja | uzasadnienie (pomiar) | zmiana kodu? |
|---|---|---|---|---|
| 1 | odrzucanie słabych próbek `--min-disp 2` | **tak** | 33% zbioru ma p95 < 2 px i uczy tożsamości; filtr usuwa je przy 1,2–1,5 losowania na próbkę | nie (jest w v2) |
| 1 | mniej próbek z klipu, więcej klipów ze sceny: 2 × ~24 zamiast 3 × 9 | **tak** | próbki z jednego klipu podobne 0,76, z różnych klipów sceny 0,49; klip wyjaśnia 89% wariancji siły RS; v1 używa tylko 20,6% klipów | nie |
| 1 | mniejszy wycinek (256–384 lub 256–512 px źródła) i T 15–30 ms | **tak** | mediana p95 +59% (wycinek), +19% (T); duchy rosną tylko przy wycinku | nie (`crop_px`) |
| 1 | proceduralny generator obrotu (wiatrak) jako drugi zbiór, z dokładnym przepływem | **tak** | w X4K tylko 0,1% próbek ma prędkość końca łopatki przy 600 obr/min; geometria myli się o ~20 px przy obrocie 72° w czasie odczytu | nowy skrypt (szkic jest) |
| 2 | przyspieszenie ×2 dla wszystkich próbek | **nie** | 66–72% próbek ×2 ma krok klatki > 2 px, czyli „duchy”; ×3 jeszcze gorzej (67%, a T ≤ 18 ms) | – |
| 2 | przyspieszenie **adaptacyjne**: ×2 tylko gdy krok przy ×2 ≤ 2 px | **tak** | szacunek: słabych 24 → 14%, mediana 7,0 → 9,4 px, duchów bez zmian (19%) | tak, ~10 linii w `make_sample` (Farnebäck między 2 klatkami wycinka) |
| 2 | zapis w `meta.json`: `flip`, `reverse`, `speedup`, krok klatki, `disp_p95` | **tak** | dziś `augment: true` nie pozwala odtworzyć próbki bez zgadywania (4 kombinacje, niejednoznaczne przy `frame_c` = 32) | tak, 3 linie |
| 2 | rozmycie ekspozycji w 20–30% próbek (średnia klatek z okna E = 2–T ms dla każdego wiersza / kolumny) | **tak, jako wariant odporności** | wiatrak w pokoju: ekspozycja 10–30 ms; w X4K < 1 ms | tak |
| 2 | szerszy sync: ±4 ms w głównym zbiorze, ±8–16 ms w osobnym wariancie (tylko ×1, T ≤ 30) | **tak** | kamery USB bez synchronizacji; przy ×2 klip ma 32 ms i sync > ±4 ms ogranicza T | nie (config) |
| 2 | więcej scen w val (np. 15–20%) albo raportowanie PSNR osobno dla klasy silnej (`strong_val.txt`) | **tak** | val ma 13 scen i jest słabszy od train (silnych 32% vs 39%); średnia zaniża efekt korekcji | nie |
| 3 | interpolacja klatek z przepływem zamiast liniowej | **może** | usunęłaby duchy i pozwoliła na ×2 wszędzie; koszt: przepływ między ~33–65 parami klatek na próbkę | tak |
| 3 | paralaksa zależna od głębi | **w generatorze proceduralnym** | w X4K nie ma głębi; w generatorze wirnik może mieć inną bazę niż tło (jak w pokoju) | tak |
| 3 | więcej niż 2 próbki z klipu, ×3, cały klip 768 px (crop 1,0) | **nie** | duplikaty treści; ×3 nie mieści T; crop 1,0 osłabia RS (p. 8 opisu zbioru: mediana 3,2 px) | – |

## 9. Proponowana komenda

Dla obecnego `src/synth_video_v2.py` (bez zmian w kodzie), konfiguracja
[`data/analiza_agent/config_v2_proponowany.json`](../data/analiza_agent/config_v2_proponowany.json):
wycinek 256–512 px źródła, T 15–30 ms, sync ±4 ms, baza 0–8 px, bez przyspieszenia (`"speedup": [1]`).

```
python src/synth_video_v2.py --src data/x4k/encoded_train --out data/rs_video_x4k_v2 \
    --config data/analiza_agent/config_v2_proponowany.json \
    --per-clip 2 --augment --clips-per-scene 24 --min-disp 2 --max-tries 4 --seed 1
```

- 24 klipy na scenę to 2181 klipów (więcej klipów nie ma w części scen), 4362 losowania. Przy odrzuceniu jak w próbie
  (8–10% próbek pomijanych, w słabszych scenach więcej) wyjdzie **ok. 3,5–3,9 tys. próbek**, wszystkie z p95 ≥ 2 px.
  Podział na train/val po scenach jak w v1 (`--val-frac 0.1`, ten sam hash nazwy sceny, więc te same sceny w val).
- Oczekiwane (próba na 40 klipach): mediana p95 ok. 13 px (v1 w tej samej próbie: 7 px), 0% słabych, ok. 74% silnych,
  PSNR wejścia ok. 21 dB.
- Gdy powstanie przyspieszenie adaptacyjne, zamienić `"speedup": [1]` na nie; ślepe `[1, 2]` z p. 9 opisu zbioru
  podnosi udział duchów do 56%.
- Wiatrak: osobny zbiór z generatora proceduralnego (rozwinięcie `skrypty/demo_wiatrak.py`), np. 2–3 tys. próbek,
  dołączony do treningu w proporcji ok. 1:1 przy dostrajaniu.

## 10. Architektura

**Rekomendacja: zostać przy RAFT + geometria (dostrojony RAFT, geometria bez wag) i nie dokładać nowej sieci,
dopóki nie ma nowych danych.** Ewentualny drugi krok to mały U-Net poprawiający wynik geometrii
(`raft_unet.py`, tryb `refine`), uczony dopiero na zbiorze v2 + generatorze obrotu.

| model | wynik na `rs_video_x4k` / wcześniej | ocena |
|---|---|---|
| bez korekcji | 26,42 dB | punkt odniesienia |
| zwykły U-Net (A, B, czasy odczytu → przepływ + residuum) | na DIV2K 20,45 dB przy wejściu 19,66 dB | nie |
| RAFT + geometria, bez treningu | 32,86 dB | działa od razu |
| RAFT + geometria, dostrojony | 34,30 dB | **najprostszy, który działa** |

Dlaczego:
1. **Problem to dopasowanie A↔B na dużych przesunięciach**, a geometria potem odwraca je dokładnie
   (`u = (d + b)/Δt`). RAFT ma objętość korelacji i iteracyjne poprawki, więc znajduje przesunięcia rzędu 10–40 px
   (95. percentyl p95 w v1: 26 px, w v2 więcej). Zwykły U-Net musiałby sam nauczyć się
   wyszukiwania dopasowań w polu recepcyjnym, a 33–44% próbek bez ruchu daje mu łatwe minimum „wyjście = wejście”.
   Tak było na DIV2K (+0,8 dB).
2. **Wiatrak łamie założenie geometrii o stałej prędkości** (obrót o 36–72° w czasie odczytu, błąd ~20 px na końcu
   łopatki). To jest właśnie miejsce dla małego U-Netu-poprawki albo dla nadzorowanego dostrojenia RAFT na generatorze
   obrotu (tam znamy dokładny przepływ). Nowa, większa architektura nic tu nie da bez danych z obrotem.

## 11. Czego nie sprawdzono

- Zgodności kierunku obrotu kamery B (`b_rot_k`) z prawdziwym zestawem: brak tu nagrań kamer; trzeba to sprawdzić
  na parze z wiatrakiem.
- Wpływu poprawek na wynik sieci (zgodnie z zasadami nic nie trenowano).
- Udziału ruchu obrotowego w X4K wprost (oceniony tylko wzrokowo na arkuszach: koła rowerów, gesty rąk).
- Warianty v2 liczone na 40 klipach (160 próbek na wariant): różnice < ~5 punktów procentowych są w granicach szumu.
