# Objaśnienia: co zrobiliśmy, dlaczego i co z tego wynika

Notatki dla mnie, do przygotowania się do rozmowy z prowadzącym. Wersja dla prowadzącego: [`podsumowanie_badan.md`](podsumowanie_badan.md).
Celowo pominięte w obu plikach: MotionNet (etap 1, inny zbiór: DIV2K i sam ruch kamery) oraz klasyczny przepływ z OpenCV (Farnebäck, DIS).

---

## 0. Dwie miary PSNR (żeby nie pomylić liczb)

W repo krążą dwa komplety liczb i trzeba wiedzieć, który jest który:

| miara | co liczy | przykłady |
|---|---|---|
| `psnr_vis` (stara, „PSNR” w notatnikach 04, 05, 07) | tylko piksele, które po korekcji mają źródło w kadrze A | wejście 26,42, RAFT gotowy 32,86, dostrojony 34,30, model główny 35,81 |
| **`psnr_full`** (tabela główna, notatnik 11) | **cały obraz**, ta sama miara dla każdej metody | wejście 26,35, RAFT gotowy 32,34, dostrojony 33,60, model główny **35,37** |

Sieci bez geometrii od początku liczyły cały obraz, więc dla nich obie miary są prawie równe. RAFT + geometria
na `psnr_vis` wyglądał lepiej, bo pomijał brzegi, których nie umie wypełnić. Dlatego w podsumowaniu jest tylko `psnr_full`.
Jeśli prowadzący pamięta „34,3 dB” albo „35,8 dB” z wcześniejszego raportu (08.11), to są te same modele, tylko w starej mierze.

Dwie tabele w podsumowaniu mają inną miarę i są tak opisane:
- podział na siłę RS: `psnr_vis`,
- zysk w funkcji |t_B − t_A|: PSNR z MSE zsumowanego po pikselach z |t_A| ≥ 5 ms (żeby patrzeć tylko na piksele, które RS naprawdę przesunął).

**Przedziały ufności.** Walidacja to 13 scen po ~27 próbek. Próbki z jednej sceny są do siebie podobne, więc bootstrap po próbkach
dawałby złudnie wąskie przedziały (±0,45 dB). Bootstrap po scenach daje uczciwe, szerokie przedziały (model główny [33,4; 37,0]),
bo sceny różnią się trudnością. Do porównań dwóch metod lepsza jest **różnica sparowana**: dla każdej próbki różnica między
metodą a modelem głównym, potem bootstrap po scenach. Trudność sceny się wtedy znosi, a przedziały różnic są wąskie i z dala od zera.

---

## 1. Co wychodzi z RAFT

**Co to jest.** RAFT to gotowa sieć do przepływu optycznego (torchvision `raft_large`, 5,26 mln wag, wstępnie wytrenowana
na publicznych zbiorach przepływu: FlyingChairs, FlyingThings, Sintel, KITTI, HD1K). Działa iteracyjnie: buduje tablicę podobieństw
każdego piksela A z każdym pikselem B (cost volume) i w 12 krokach poprawia oszacowanie.

**Co zwraca.** Pole przepływu **d(p)** o rozmiarze 2×256×256: dla każdego piksela *p* obrazu A wektor (dx, dy), czyli gdzie ten sam
punkt sceny jest w obrazie B (B obrócony z powrotem do orientacji A). **RAFT nie wie nic o czasie ani o rolling shutterze.**
Odpowiada tylko na pytanie „gdzie jest ten punkt w drugim obrazie”. Zamianą tego na korekcję zajmuje się geometria.

**Wyniki (psnr_full):**
- **gotowy RAFT + geometria v1: 32,34 dB** (wejście 26,35), czyli **+6 dB bez żadnego uczenia na naszych danych**;
- z geometrią v2: 32,81 dB;
- **dostrojony (10 epok) + geometria v1: 33,60 dB**; dostrojony dalej z v2 (+5 epok): 33,89 dB.

**Jak go dostrajamy.** Geometria jest zbudowana z różniczkowalnych operacji (dzielenie, `grid_sample`), więc gradient
z błędu obrazu (L1 wynik vs wzorzec GS) płynie przez geometrię do wag RAFT. Uczymy tylko RAFT, geometria jest stała.
lr 2·10⁻⁵, bs 8, zamrożone BatchNorm, przycinanie gradientu.

**Dlaczego dostrajanie pomaga i dlaczego szybko się nasyca.** RAFT uczono na parach zwykłych zdjęć, gdzie oba obrazy są „sztywne”.
U nas oba obrazy są zniekształcone, każdy w innym kierunku (A pochyla w pionie, B w poziomie), do tego szum. Dostrojenie uczy go
tego typu par i tego, gdzie błąd przepływu najbardziej psuje obraz. Zysk ok. +1,3 dB, nasycenie po ok. 5 epokach,
dalsze 5 epok z v2 dodało już tylko ~0,07 dB. Wniosek: RAFT dobrze dopasowuje punkty już na starcie; zostają błędy, których
sam przepływ nie naprawi (pas na antyprzekątnej, zasłonięcia), i tym zajmuje się U-Net.

---

## 2. Jak działa algorytm geometrii (`src/raft_geom.py`)

### Czasy odczytu

Obraz 256×256, czas odczytu klatki *T* (10–30 ms), *t* = 0 to środek odczytu kamery A.
- Kamera A czyta wiersze: piksel w wierszu *y* jest zarejestrowany w chwili **t_A = (y/255 − 0,5)·T**.
- Kamera B jest obrócona, więc w kadrze A czyta kolumny (od prawej): **t_B = sync + ((255 − x)/255 − 0,5)·T**.

Każdy piksel ma więc dwie znane chwile: kiedy widziała go kamera A, a kiedy kamera B.

### Krok 1: prędkość z przepływu

Punkt sceny porusza się z prędkością *u* [px/ms]. W chwili 0 jest w miejscu *q*.
- Kamera A widzi go w chwili t_A w miejscu p_A = q + u·t_A.
- Kamera B widzi go w chwili t_B w miejscu p_B = q − b + u·t_B (*b* = baza, czyli przesunięcie między kamerami).

Odejmując: d = p_B − p_A, czyli **d + b = u·(t_B − t_A)**, więc

> **u = (d + b) / (t_B − t_A)**

RAFT daje *d*, czasy znamy z numerów wiersza i kolumny, bazę znamy z kalibracji. Dzielimy i mamy prędkość każdego piksela.

**Przykład (w zaokrągleniu):** T = 20 ms, piksel (x = 191, y = 191). Kamera A czyta go w t_A ≈ +5 ms, kamera B w t_B ≈ −5 ms,
czyli 10 ms wcześniej. Obiekt jedzie w prawo 1 px/ms, więc w obrazie B jest 10 px bardziej w lewo: d = −10 px.
u = −10 / (−5 − 5) = +1 px/ms. W chwili 0 punkt był w x = 191 − 1·5 = 186. Tam trafi jego kolor w obrazie wynikowym.

### Krok 2: pas na antyprzekątnej

Gdy t_B ≈ t_A, obie kamery widzą punkt w tej samej chwili, więc d ≈ b niezależnie od ruchu i **prędkości nie da się zmierzyć**
(dzielenie przez ~0). Przy sync = 0 dzieje się to dla y = 255 − x, czyli na **antyprzekątnej** kadru (od lewego dołu do prawego góry).

Rozwiązanie:
- pewność **c = (Δt² / (Δt² + ε²))²**, gdzie ε = 0,05·T (1 ms przy T = 20). Daleko od antyprzekątnej c ≈ 1, na niej c ≈ 0;
- prędkość w pasie (szerokim na ~25 px) jest średnią prędkości **pewnych** sąsiadów w oknie 61×61 px, ważoną przez *c*;
- u = c·u_zmierzona + (1 − c)·u_z_sąsiedztwa.

To jest główne źródło błędów w środku kadru. Widać je też w ablacji: zysk modelu przy |t_B − t_A| ≈ 1 ms wynosi +6,8 dB,
a przy 7,5–10,5 ms +13,7 dB.

### Krok 3: cofnięcie obrazu do chwili 0 (iteracja punktu stałego)

Chcemy dla każdego piksela wyjściowego *q* znaleźć, skąd wziąć kolor w obrazie A. Zachodzi **p = q + u(p)·t_A(p)**.
Problem: *p* jest po obu stronach równania. Prędkość znamy w miejscu, gdzie punkt był widziany (p), a nie tam, gdzie ma trafić (q),
a czas t_A zależy od wiersza *p*. Dlatego iterujemy: start p = q, potem 6 razy p ← q + u(p)·t_A(p). Na koniec `grid_sample`
pobiera kolor A z *p* (interpolacja dwuliniowa). Piksele, dla których *p* wypada poza kadr A, nie mają źródła.
To są brzegi, które v1 zostawia błędne, a U-Net bierze z kamery B.

### Geometria v2 (`correct_ab`): co poprawia

Dwa błędy v1 na krawędziach ruchomych obiektów:
1. **Na krawędzi obiektu równanie ma dwa rozwiązania**: tło (u = 0, p = q) i obiekt. v1 startuje z p = q, więc zostaje na tle i zostawia
   pasek tła tam, gdzie powinien być obiekt. v2 startuje dwa razy: z p = q i z prędkości najszybszego sąsiada (okno 31 px).
   Jeśli oba rozwiązania są spójne, wybiera szybsze, bo obiekt w ruchu zasłania tło.
2. **Tło, które obiekt zasłaniał akurat wtedy, gdy A czytała te wiersze**, nie ma w A żadnego spójnego rozwiązania: A go nie widziała.
   v2 bierze te piksele z kamery B, cofniętej do chwili 0 tak samo. Do tego potrzebna jest prędkość na siatce pikseli B, więc RAFT
   liczy się drugi raz, B → A.

Na całym zbiorze v2 daje mało (+0,47 dB przy gotowym RAFT), bo poprawia tylko krawędzie ruchomych obiektów, a ~1/3 próbek
prawie nie ma ruchu. Na syntetycznym kole widać to wyraźnie (26,9 → 29,1 dB, `psnr_vis`).

### Założenia geometrii (ograniczenia)

- ruch jednostajny po prostej w czasie odczytu (szybki obrót, np. wiatrak, łamie to założenie),
- znane T, sync i baza (w danych syntetycznych znamy je dokładnie, na prawdziwym stanowisku trzeba je zmierzyć),
- punkt musi być widoczny w obu kamerach (inaczej nie ma przepływu).

---

## 3. Model główny: RAFT + geometria + U-Net

Geometria daje dwa obrazy cofnięte do chwili 0: z kamery A i z kamery B. U-Net (2,7 mln wag) dostaje oba, prędkość i czasy, i zwraca:
- poprawki obu pól przesunięć (da, db), po kilka pikseli,
- maskę mieszania A/B dla każdego piksela,
- drobną poprawkę jasności (residuum).

**wynik = maska·warp(A, przesunięcie_A + da) + (1 − maska)·warp(B, przesunięcie_B + db) + residuum**

Na starcie maska ≈ 0,88 dla A i małe wagi głowicy, więc model zaczyna od wyniku samej geometrii i tylko go poprawia.
Uczenie: 30 epok U-Netu przy zamrożonym RAFT (35,24 dB), potem 10 epok razem z RAFT (35,37 dB, +0,13 dB, czyli już nasycenie).

**Co naprawia (+1,45 dB [1,28; 1,65], lepiej w 303/321):**
- szew między obszarami z kamery A i z kamery B (v2 bierze piksele z B „na twardo”, maska robi to płynnie),
- brzegi kadru, których A nie widziała: średnia waga B w masce rośnie z 0,12 na starcie do ~0,4,
- poszarpane krawędzie obiektów i drobne błędy przepływu w pasie antyprzekątnej,
- rozmycie interpolacji, różnicę wzmocnienia między kamerami, szum.

Uwaga do pytań: szum jest dodany tylko do wejścia, wzorzec jest czysty, więc część zysku U-Netu może być zwykłym odszumianiem.
Zysk U-Netu jest podobny w każdej klasie siły RS (+1,3 do +1,6 dB), czyli nie zależy od tego, jak silny był RS.

---

## 4. Modele bez geometrii (autoenkoder, U-Net, `corr`)

Pytanie z karty tematu: czy sieć sama nauczy się korekcji („bezpośrednia synteza klatek”), bez jawnej geometrii?
Wszystkie trzy dostają 10 kanałów: A, B, współrzędne (x, y) i mapy czasów odczytu obu kamer. Mają więc tę samą informację co geometria.

### Autoenkoder bez skrótów (`ae`): 10,2 mln wag, **23,27 dB** (gorzej niż wejście 26,35)

- Enkoder zmniejsza obraz 4 razy o połowę: 256 → 128 → 64 → 32 → 16 px. Wąskie gardło (latent) ma 16×16×512 = 131 tys. liczb
  (para wejściowa to 196 tys.).
- Dekoder odtwarza z tego cały obraz 256×256. **Nie ma połączeń skrótowych**, więc wszystko, co ma być na wyjściu, musi przejść
  przez siatkę 16×16, czyli jedną komórkę na 16×16 px.
- Skutek: drobne tekstury i ostre krawędzie się gubią i obraz wychodzi rozmyty. Rozmycie kosztuje w PSNR więcej, niż daje korekcja RS,
  więc wynik jest gorszy od nieskorygowanego wejścia.
- Rola w pracy: punkt odniesienia „co daje samo wąskie gardło”. Pokazuje, że do tego zadania trzeba przenosić szczegóły z wejścia.

### Zwykły U-Net: 2,72 mln wag, 28,42 dB (30 ep.) → **30,94 dB (200 ep.)**

- To też enkoder-dekoder, ale **ze skrótami**: na każdej rozdzielczości cechy enkodera idą prosto do dekodera, więc szczegóły nie giną.
- Nie syntetyzuje obrazu od zera. Przewiduje przesunięcie każdego piksela A i poprawkę jasności. Wynik to przesunięte piksele A,
  dlatego jest ostry (najwyższy SSIM wśród sieci bez geometrii: 0,895).
- Plateau od ok. 100. epoki.

### Enkoder + korelacja (`corr`): 6,24 mln wag, **31,06 dB (80 ep.)**

- A i B przechodzą przez **ten sam** enkoder (sieć syjamska), cechy na 1/2, 1/4 i 1/8 rozdzielczości.
- **Warstwa korelacji (cost volume)** porównuje cechy A z cechami B przesuniętymi o ±4 komórki na 1/8 (±32 px) i ±3 na 1/4 (±12 px).
  Daje sieci gotowe porównanie „gdzie w B jest ten fragment A”, czyli to, czego zwykły U-Net musi się domyślić sam.
  To ta sama idea co w RAFT, ale bez iteracji i bez pretreningu.
- Dekoder ze skrótami, wynik = A + poprawka obrazu (bez przesuwania pikseli, czyli „domalowuje”). Stąd niższy SSIM (0,887) przy podobnym PSNR.
- Do 31 dB dochodzi szybciej niż U-Net (80 epok zamiast ~100), dalej też stoi.
- **Niestabilny numerycznie**: sieć nie ma warstw normalizacji i przy lr 3·10⁻⁴ jej aktywacje przekraczały zakres fp16 (NaN).
  Po przejściu na bf16 zmienił się tylko objaw (wyjście nasycone na czarny obraz). Dwie z trzech prób uczenia rodziny `corr` się wywróciły,
  więc wariant `corr` bez kamery B i `corrgeom` (ta sama sieć przewidująca przepływ dla geometrii) porzuciliśmy. Ablację kamery B niesie para U-Netów.

### Wspólny wniosek

Trzy różne architektury zatrzymały się na ≤ 31 dB, a gotowy RAFT z geometrią **bez żadnego uczenia na naszych danych** daje 32,34–32,81.
Po dostrojeniu RAFT 33,60–33,89.

---

## 5. Dlaczego więcej parametrów nic by nie zmieniło

**Uczciwie: tego nie przetestowaliśmy wprost.** Zaplanowany test `corr_w64` (szerokość 64, ~25 mln wag, ok. 4× więcej) został pominięty
jako niepotrzebny do tezy. To jest **wniosek pośredni** z kilku obserwacji. Jeśli prowadzący zapyta, mówić „wszystko wskazuje, że…”, a nie „sprawdziliśmy”.

Argumenty:

1. **Liczba wag nie idzie w parze z wynikiem.** Najwięcej wag ma autoenkoder (10,2 mln) i jest najgorszy (23,27). `corr` (6,24 mln)
   i U-Net (2,72 mln, ponad 2× mniej) kończą prawie tak samo (31,06 vs 30,94). RAFT + geometria (5,26 mln, mniej niż `corr`) jest wyżej.
   O wyniku decyduje **budowa** (skróty, dopasowanie punktów, jawna geometria), nie rozmiar.

2. **To nie jest niedouczenie.** U-Net ma na koniec stratę treningową 0,0113, **niższą niż `corr` przy tym samym wyniku na walidacji**.
   Sieć dobrze dopasowuje się do zbioru treningowego, a nie przenosi tego na nowe sceny. Więcej wag przy 2397 próbkach z 97 scen
   to raczej większe przeuczenie, nie lepsza generalizacja.

3. **Plateau przy długim uczeniu.** U-Net: 30 → 200 epok (6,7× więcej kroków) dało +2,5 dB, potem płasko od ~100. epoki. `corr`: +0,1 dB
   w ostatnich 20 epokach. Model, któremu brakuje pojemności, zwykle dalej powoli schodzi ze stratą; tu walidacja stoi.

4. **Najmocniejszy argument: ablacja |t_B − t_A|.** Kluczowa operacja to *przesunięcie / różnica chwil odczytu*. Model z geometrią
   ma jej zysk rosnący z Δt (+6,8 → +13,7 dB), bo jawnie dzieli przez Δt. Zysk zwykłego U-Netu z kamery B **lekko maleje** z Δt
   (+5,2 → +3,7 dB), a jego mapa zysku zależy od wiersza (od |t_A|), nie od przekątnej. Sieć nauczyła się „B to drugi, inaczej
   zniekształcony widok”, a nie „B to pomiar w innej chwili, więc da się policzyć prędkość”. To ograniczenie sposobu, w jaki sieć rozwiązuje
   zadanie, a nie jej rozmiaru. Szersze warstwy nie dodają mechanizmu dzielenia przez zmienną w przestrzeni różnicę czasów ani
   wyszukiwania odpowiedników na 10–40 px.

5. **Autoenkoder: problem to kształt, nie rozmiar.** Szersze warstwy nadal przepuszczają wszystko przez siatkę 16×16, więc szczegóły
   i tak by się gubiły. Pomogłoby dodanie skrótów, ale wtedy to już jest U-Net (który ma 30,94).

6. **Kontrprzykład z drugiej strony:** U-Net o tym samym rozmiarze (2,7 mln), ale **położony na geometrii** (model główny), dodaje +1,45 dB
   ponad 33,9 dB. Ta sama liczba wag daje 35,37 dB, gdy dostanie gotowy mechanizm, i 30,94 dB, gdy musi go wymyślić sam.

Gdyby prowadzący chciał dowodu wprost: uruchomić `corr_w64` lub szerszy U-Net (200 epok, L4). Przewidywanie: ok. 31 dB.

---

## 6. Ablacja kamery B: jak ją czytać

- **Para U-Netów (200 epok, te same ustawienia):** z kamerą B 30,94 dB, z B zastąpioną kopią A 26,51 dB (wejście 26,35).
  Bez B sieć prawie nic nie poprawia. Z jednego obrazu RS nie da się odróżnić pochylonego obiektu od pochylenia przez RS, bo brakuje
  drugiej chwili pomiaru.
- **Analiza bez uczenia (notatnik 09):** zysk dla pikseli pogrupowanych według |t_B − t_A|. Model z geometrią zyskuje najwięcej
  przy 7,5–10,5 ms różnicy, najmniej przy ~1 ms (antyprzekątna). To dokładnie to, co przewiduje wzór u = (d + b)/Δt:
  im większa Δt, tym dokładniej mierzymy prędkość. **To jest eksperymentalne potwierdzenie sensu prostopadłego ustawienia kamer.**
  Przy dużych Δt (14–25 ms) zysk lekko spada (+11,5–11,8 dB). Przyczyny nie sprawdzaliśmy; takie piksele leżą blisko rogów kadru,
  gdzie częściej brakuje źródła w A, więc to prawdopodobne wyjaśnienie, ale niepotwierdzone.

---

## 7. Kolejne decyzje w skrócie (chronologicznie)

| kiedy | decyzja | dlaczego |
|---|---|---|
| 24.10 | zbiór z nagrań 1000 fps (`rs_video_x4k`) zamiast zdjęć z ruchem kamery | cel to ruch obiektów (wiatrak); z nagrań każdy wiersz można wziąć z prawdziwej klatki z chwili odczytu |
| 24.10 | RAFT + stała geometria zamiast sieci uczonej od zera | gotowy, dobry przepływ + znana fizyka odczytu; od razu +6 dB bez uczenia |
| 24.10 | dostrajanie RAFT przez geometrię (strata na obrazie) | nie mamy prawdziwego przepływu jako wzorca, mamy obraz GS; geometria różniczkowalna |
| 03.11 | geometria v2 | demo na kole pokazało pasek tła na krawędzi i dziury za obiektem |
| 03.11 | zwykły U-Net jako punkt odniesienia | pokazać, ile daje sama sieć bez geometrii |
| 08.11 | U-Net na końcu (model główny) | po geometrii zostają szew A/B, brzegi kadru, krawędzie; nasycenie samego RAFT |
| 11.11 | autoenkoder i `corr` (synteza wprost) | karta tematu wymaga porównania z bezpośrednią syntezą; `corr` daje sieci dopasowanie punktów |
| 11.11 | 200 epok dla modeli od zera | żeby nikt nie zarzucił, że sieci bez geometrii były za krótko uczone |
| 14.11 | jedna miara (`psnr_full`), CI po scenach, różnice sparowane | stare liczby mieszały dwie miary; 13 scen wymaga uczciwej niepewności |
| 14.11 | porzucone `corr_single`, `corrgeom`, testy lr i szerokości | sieć `corr` niestabilna; teza i tak stoi na parze U-Netów i analizie Δt |
| 15.11 | zbiór rozszerzony (X4K + warstwowy), ta sama liczba kroków | sprawdzić, czy więcej danych pomaga syntezie bardziej niż geometrii; **wyników jeszcze nie ma w repo** |

---

## 8. Pytania, które mogą paść

- **„Czemu nie wystarczy większa sieć?”** → sekcja 5. Kluczowe: ablacja Δt i ten sam U-Net 2,7 mln: 30,94 bez geometrii, 35,37 na geometrii.
- **„13 scen to mało, czy różnice są pewne?”** → różnice sparowane względem modelu głównego mają CI95 z dala od zera dla każdej metody.
  Bezwzględne liczby mają szerokie przedziały, bo sceny różnią się trudnością.
- **„Ile daje sama geometria, a ile pretrening RAFT?”** → tego nie rozdzieliliśmy w pełni; własna sieć przepływu (`corrgeom`) była niestabilna.
  To jest otwarte i jest w ograniczeniach.
- **„Czy U-Net nie jest odszumiaczem?”** → częściowo może być (szum tylko na wejściu). Ale widać też efekty, których odszumianie nie da:
  zniknięcie szwu A/B i brzegi z kamery B (waga B w masce ~0,4).
- **„Co z antyprzekątną?”** → tam Δt ≈ 0, prędkość uzupełniana z sąsiedztwa; najmniejszy zysk (+6,8 dB przy ~1 ms). Przy prawdziwych
  kamerach można by przesunąć start kamery B (sync), żeby pas wypadł poza ważną część kadru.
- **„Szybkość?”** → model główny 56 ms na parę 256×256 na L4 (dominuje RAFT, 12 iteracji); sieci bez geometrii 4–11 ms.
- **„Czy działa na prawdziwych kamerach?”** → jeszcze nie sprawdzone ilościowo (brak wzorca GS); demo z wiatrakiem w planach.
