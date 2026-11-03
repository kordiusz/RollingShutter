## Streszczenie prac
Do uczenia i oceny powstał zbiór `rs_video_x4k` z nagrań 1000 fps. Na zbiorze walidacyjnym metoda poprawia PSNR
z 26,4 dB (wejście) do 32,9 dB bez treningu i do **34,3 dB** po dostrojeniu RAFT.

## Zbiór danych `rs_video_x4k`

Źródłem jest **X4K1000FPS** (Sim i in., XVFI, ICCV 2021 - https://arxiv.org/abs/2103.16206): nagrania 1000 fps, część treningowa to klipy 65 klatek
(64 ms) o rozmiarze 768×768 px ze 110 scen plenerowych (licencja: badania i dydaktyka). Skrypt `src/synth_video.py`:
1. wybiera wycinek 384–768 px (`crop_frac` 0,5–1,0) i zmniejsza go do 256×256; dla kamery B ten sam wycinek
   jest przesunięty o bazę 0–8 px;
2. losuje czas odczytu T = 10–30 ms i przesunięcie startu kamery B (sync) ±2 ms;
3. składa obraz RS: wiersz y kamery A pochodzi z klatki nagranej w chwili `(y/255 − 0,5)·T`, a kolumna x kamery B
   z chwili `sync + ((255 − x)/255 − 0,5)·T`, z liniową interpolacją między sąsiednimi klatkami (1 ms);
4. jako wzorzec GS bierze dokładną klatkę ze środka odczytu A;
5. dodaje wzmocnienie 0,9–1,1, szum σ ≤ 0,01 oraz losowe odbicie i odwrócenie czasu.

Wybrano do 9 klipów na scenę i po 3 próbki na klip: **2718 próbek z 906 klipów**, z podziałem po scenach:
**2397 train / 321 val** (4 obrazy - RS A, RS B i GS A, GS B z czego tutaj tylko jeden obraz GS jest nam potrzebny)

## Jak działa korekcja

Kamera A czyta obraz wierszami od góry do dołu, a kamera B kolumnami, więc ten sam punkt sceny każda kamera
fotografuje w innej chwili, a czas między tymi chwilami znamy z budowy kamer. Gotowa sieć **RAFT** porównuje oba obrazy
i dla każdego piksela mówi, o ile punkt przesunął się między obrazem A a obrazem B. Przesunięcie podzielone przez znaną
różnicę czasu daje **prędkość** każdego punktu. Znając prędkość i chwilę, w której kamera A sfotografowała dany wiersz,
cofamy każdy punkt tam, gdzie był w jednej wspólnej chwili. Tak powstaje obraz bez rolling shuttera. Ta część to zwykłe
obliczenia, bez uczenia. Problem pojawia się tylko tam, gdzie obie kamery czytają punkt prawie w tej samej chwili:
różnica czasu jest bliska zera, więc prędkości nie da się policzyć i jest uzupełniana z sąsiednich pikseli.

## Trening

RAFT był już wytrenowany przez autorów do śledzenia ruchu między dwiema klatkami filmu. Na naszym zbiorze dostroiliśmy go tak, żeby obraz
po korekcji był jak najbliższy wzorcowi; obliczenia czasu i prędkości się nie uczą. Trening trwał 10 epok. Sam gotowy RAFT poprawia obraz o 6,4 dB, dostrojenie dodaje kolejne 1,4 dB.

(wyniki z colaba)
| model (val, 321 próbek) | PSNR [dB] | SSIM |
|---|---|---|
| bez korekcji (RS A) | 26,42 | 0,795 |
| RAFT gotowy + geometria | 32,86 | 0,936 |
| RAFT dostrojony + geometria | **34,30** | **0,941** |

## Wnioski i dalsze kroki

Geometria daje większość zysku już bez uczenia. Średni wynik jest jednak zaniżony przez 1/3 próbek prawie bez ruchu,
więc wyniki trzeba raportować w podziale na siłę RS (tym teraz sie zajmuje).