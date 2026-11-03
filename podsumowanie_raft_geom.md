# Korekcja rolling shutter z dwóch prostopadłych kamer: zbiór danych i RAFT + geometria

*Stan prac: 2026-10-25*

## Streszczenie prac

ozwiązanie łączy gotową sieć przepływu optycznego **RAFT** ze
**stałą geometrią czasu odczytu**: RAFT znajduje odpowiadające sobie punkty w obrazach A i B, a geometria zamienia to
przesunięcie na prędkość każdego piksela i cofa obraz A do jednej chwili. Metoda koryguje ruch kamery i ruch obiektów.
Do uczenia i oceny powstał zbiór `rs_video_x4k` z nagrań 1000 fps. Na zbiorze walidacyjnym metoda poprawia PSNR
z 26,4 dB (wejście) do 32,9 dB bez treningu i do **34,3 dB** po dostrojeniu RAFT.

## Zbiór danych `rs_video_x4k`

Źródłem jest **X4K1000FPS** (Sim i in., XVFI, ICCV 2021): nagrania 1000 fps, część treningowa to klipy 65 klatek
(64 ms) o rozmiarze 768×768 px ze 110 scen plenerowych. Skrypt działał następująco:
1. wybiera wycinek 384–768 px (`crop_frac` 0,5–1,0) i zmniejsza go do 256×256; dla kamery B ten sam wycinek
   jest przesunięty o bazę 0–8 px ( bo fizyczne kamery sa w odleglosci od siebie)
2. losuje czas odczytu T = 10–30 ms i przesunięcie startu kamery B (sync) ±2 ms;
3. składa obraz RS: wiersz y kamery A pochodzi z klatki nagranej w chwili `(y/255 − 0,5)·T`, a kolumna x kamery B
   z chwili `sync + ((255 − x)/255 − 0,5)·T`, z liniową interpolacją między sąsiednimi klatkami (1 ms);
4. jako wzorzec GS bierze dokładną klatkę ze środka odczytu A;
5. dodaje wzmocnienie 0,9–1,1, szum σ ≤ 0,01 oraz losowe odbicie i odwrócenie czasu.

Wybrano do 9 klipów na scenę i po 3 próbki na klip: **2718 próbek z 906 klipów**, z podziałem po scenach:
**2397 train (97 scen) / 321 val (13 scen)**. 

## Algorytm geometrii

RAFT (torchvision `raft_large`, 12 iteracji) zwraca przepływ d(p) z obrazu A do obrazu B obróconego z powrotem do
orientacji A. Punkt widziany przez A w chwili t_A(p) i przez B w chwili t_B(p + d) przesunął się o d + b, gdzie b to
baza, więc jego prędkość wynosi **u = (d + b) / (t_B − t_A)** [px/ms]. Na przekątnej, gdzie obie kamery czytają punkt
w tej samej chwili (Δt ≈ 0), prędkość jest nieokreślona i zostaje uzupełniona średnią z wiarygodnych sąsiadów
(waga c = (Δt² / (Δt² + ε²))², ε = 0,05·T, okno 61 px). Obraz wynikowy w chwili t = 0 powstaje przez próbkowanie A
w punkcie p spełniającym p = q + u(p)·t_A(p) (6 iteracji punktu stałego, `grid_sample`). Cała operacja jest
różniczkowalna (`src/raft_geom.py`). Z dokładnym przepływem daje ok. 36 dB, co jest górną granicą samej geometrii.

## Trening RAFT

Uczony jest tylko RAFT (wstępnie wytrenowany na zbiorach przepływu C+T+S+K+H), a geometria pozostaje stała. Funkcja
straty to L1 między wynikiem a wzorcem GS w pikselach, których źródło leży w kadrze A. Ustawienia: AdamW, lr 2·10⁻⁵
z harmonogramem OneCycle, batch 8, 10 epok, zamrożone BatchNorm, przycinanie gradientu 1,0 (`src/train_raft.py`,
`notebooks/04_raft_geom.ipynb`, Colab Pro, 108 s/epokę). Dostrojenie dodaje 1,4 dB, a wynik nasyca się po około
5 epokach.

| model (val, 321 próbek) | PSNR [dB] | SSIM |
|---|---|---|
| bez korekcji (RS A) | 26,42 | 0,795 |
| RAFT gotowy + geometria | 32,86 | 0,936 |
| RAFT dostrojony + geometria | **34,30** | **0,941** |

