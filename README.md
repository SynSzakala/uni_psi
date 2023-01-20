Jakub Walusiak, Juliusz Kościołek, Marcin Dohnalik

# Bot do gry Snake przy użyciu uczenia przez wzmocnienie (Reinforcement Learning)

## Pierwsze podejście: Splendor

Początkowym celem projektu było stworzenie bota grającego w grę karcianą.
Splendor jest karcianą grą strategiczną opartą na zbieraniu zasobów i wykorzystywaniu ich do dalszej ekspansji.
Uznaliśmy, że będzie ona idealnym kandydatem na wytrenowanie bota, ze względu na poniższe cechy:

- Stosunkowo proste zasady: Trenowanie bota wymaga zaimplementowania reguł środowiska, w którym będzie się on
  poruszał; im są one prostsze, tym łatwiej.
- Dość niski poziom losowości: Wysoka entropia zmniejsza skuteczność inteligentnych taktyk, bardziej premiuje proste
  szacowanie prawdopodobieństwa (przykład: wojna). Zbyt duży determinizm ogranicza zmienność gier i zmusza do
  wypracowania "jedynej słusznej" taktyki (przykład: kółko i krzyżyk)
- Emergentne mechaniki: Z prostych zasad wynikają ciekawe zależności, otwierając drogę całej gamie ciekawych i
  nieoczywistych strategii, dajemy możliwość znalezienia takiej botowi.

### Uczenie przez wzmocnienie

Przy trenowaniu botów, uczenie przez wzmocnienie jest oczywistym wyborem. Polega ono na umieszczeniu agenta w środowisku
o zdefiniowanych regułach, a następnie pozwolenie mu na wykonywanie akcji na podstawie obserwacji tego środowiska.
Środowisko może następnie karać bądź nagradzać takiego agenta za wykonane akcje. Takie podejście ma swoje plusy i
minusy:

- Do trenowania nie jest potrzebny gotowy zestaw danych — wystarczy zaimplementować zasady działania środowiska. Agent
  poniekąd sam zbiera dane, od początkowej eksploracji reguł poprzez losowe akcje, aż po tworzenie skomplikowanych
  strategii w celu maksymalizowania nagrody.
- Agent zainteresowany jest wyłącznie maksymalizowaniem nagrody (albo minimalizowaniem kary) - źle zaprojektowana
  funkcja nagrody może powodować wykształcenie niepożądanych
  zachowań ([przykład OpenAI](https://openai.com/blog/faulty-reward-functions/)).

### Self-Play

Klasyczne uczenie przez wzmocnienie bierze pod uwagę tylko agenta i środowisko, w którym działa. W jaki sposób
wykorzystać to podejście w przypadku systemów, w których agenci rywalizują ze sobą?
Rozwiązaniem tego problemu jest podejście Self-Play, zastosowane między innymi w
projekcie [Google AlphaZero](https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go).
Polega ono na trenowaniu aktualnej wersji agenta, ze swoimi poprzednimi wersjami jako przeciwnikami. Agent znajduje nowe
taktyki pozwalające pokonać szeroką gamę graczy.
Framework [`SIMPLE`](https://github.com/davidADSP/SIMPLE) pozwala na prostą implementację tego podejścia dla gier o
dowolnych zasadach.

![img_3.png](img_3.png)

### Gym Framework

Standardowym narzędziem do implementacji własnych środowisk dla uczenia przez wzmocnienie
jest [Gym Framework](https://github.com/openai/gym).
Główne elementy środowiska to:

- Przestrzeń obserwacji: Zbiór wszystkich obserwacji (potencjalnie nieskończony), których agent może dokonać w trakcie
  gry. Przykładowo, dla szachów
  będzie zawierał on wszystkie możliwe ułożenia figur na szachownicy.
- Przestrzeń akcji: Skończony zbiór wszystkich akcji, które agent może dokonać w trakcie gry (nie wszystkie muszą być
  zawsze dozwolone). Przykładowo, dla gry Minesweeper będzie on zawierał pozycje wszystkich pól.
- Funkcja nagrody: Na podstawie wykonanej akcji aktualizuje stan środowiska, zwraca kolejną obserwację oraz nagrodę.
- Funkcja maskująca akcje: Na podstawie aktualnego stanu środowiska zwraca aktualnie dozwolony podzbiór przestrzeni
  akcji.

### Dlaczego się nie udało?

Pod koniec implementacji mechaniki gry uświadomiliśmy sobie, że nie będziemy w stanie łatwo zintegrować jej z
paradygmatem RL. Przestrzenie obserwacji i akcji, które zaczęliśmy projektować, wydawały się bardzo skomplikowane i
ciężkie do normalizacji. Ze
względu na specyfikę gry, przestrzeń akcji mogłaby okazać się nieskończona. W tych okolicznościach uznaliśmy, że szansa
na niepowodzenie projektu jest zbyt duża i postanowiliśmy wybrać prostszy problem.

## Drugie podejście: Snake

Wyciągając wnioski z próby wytrenowania bota do gry karcianej dla wielu graczy, postanowiliśmy wykorzystać środowisko
reprezentujące typowy problem kontroli jednego agenta. Zdecydowaliśmy się na grę Snake z powodów podobnych do tych, dla
których wybraliśmy wcześniej Splendor oraz biorąc pod uwagę poprzednie doświadczenia:

- Proste zasady
- Losowość na poziomie pozwalającym na wypracowanie skutecznych strategii
- Szybko rosnący poziom trudności: Zebranie kilku pierwszych punktów jest proste, dłuższy ogon wymaga już planowania jak
  się z nim nie zderzyć.
- Jasna funkcja nagrody: Zdobycie punktu lub śmierć.
- Jasna przestrzeń obserwacji: Stan planszy.
- Jasna przestrzeń akcji: Ruch w dowolnym z 4 kierunków.

### Konstrukcja środowiska

Implementacja mechaniki gry Snake jest relatywnie prosta, jednak konstrukcja samego środowiska wymagała podjęcia decyzji
ważnych dla późniejszego treningu.

Po pierwsze, zła funkcja nagrody może spowodować niepożądane zachowania. Zbyt duże nagrody mogą powodować podejmowanie
niepotrzebnego ryzyka dla zdobycia punktu, nawet jeśli kończy się to przegraną. Zbyt niskie nagrody mogą znowu powodować
pasywność.
Zaprojektowaliśmy funkcję nagrody w następujący sposób:

1. Zdobycie punktu: 1
2. Przegrana: -10
3. Każda inna akcja: -0.01

Kluczowy jest stosunek kary 3 do nagrody 1. Definiuje on, jak bardzo karana jest pasywność. Początkowe ustawienia
oznaczają, że agent może wykonać do 100 kroków pomiędzy zdobyciami punktu, żeby wychodzić "na plus".

Po drugie, nasuwają się dwa podejścia co do przestrzeni obserwacji:

#### 1. "Lokalne"/"Subiektywne"

Agent wie, czy w danym kierunku znajduje się punkt, czy przeszkoda (ściana/ogon). To podejście wydaje się popularne
w [publicznie dostępnych materiałach dla początkujących](https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36).

Zalety:

- Niewielka, niezależna od wielkości planszy przestrzeń obserwacji.
- Szybka nauka podstawowych zasad (poruszanie się w stronę punktu, unikanie przeszkody) dzięki praktycznie
  bezpośredniemu mapowaniu nagroda/kara kontra kierunek zawartemu w obserwacji.

Wady:

- Zbytnie ułatwienie zadania, przerzucenie większości faktycznej logiki na implementację środowiska.
- Brak szerszej świadomości sytuacji ogranicza możliwości agenta; powszechna podatność agenta do otaczania przez własny
  ogon.

![](enclosing.gif)

#### 2. "Globalne"/"Obiektywne"

Agent widzi stan wszystkich pól planszy (tak jak człowiek grający w grę).

Zalety:

- Pełna świadomość sytuacji, możliwość zapobiegania okrążeniu przez własny ogon.
- Nie ma konieczności implementacji dodatkowej logiki przez środowisko.

Wady:

- Rozmiar obserwacji zależny od rozmiaru planszy, zmiana rozmiaru wymaga wytrenowania nowej sieci (lub wykorzystania
  uczenia transferowego)

Możliwa jest oczywiście modyfikacja obu tych podejść w celu eliminacji wad. Przykładowo, podejście "globalne", ale z
widocznością ograniczoną do `x` pól w każdą stronę od głowy eliminuje ograniczenie stałego rozmiaru, ale daje
teoretyczną szansę na otoczenie przez ogon, gdy jest on ma długość większą niż `x^2` i jest w całości poza obszarem
obserwacji.

Wybraliśmy podejście 2, ponieważ ciekawiło nas, o ile lepszy efekt możemy w ten sposób osiągnąć.
Obserwacja została znormalizowana do postaci `wysokość*szerokość*4*{0;1}` (4-elementowa tablica z wartością `1` na
pozycji odpowiadającej stanowi pola, dla każdego z pól).

### Algorytm uczenia

Trening wymaga wybrania algorytmu uczącego. Wiele z nich jest publicznie dostępnych, ale te najczęściej używane
dzielą się na dwie kategorie.

#### Q-Learning

Algorytmy klasy Q-Learning bazują na aproksymacji funkcji, która wylicza oczekiwaną nagrodę (aktualną i przyszłą) na
podstawie obserwacji i danej akcji. Jeśli znamy wartości tej funkcji dla każdej z dostępnych akcji, wystarczy wybrać tą
z największą wartością.

#### Policy Gradient

Algorytmy klasy Policy Gradient przyjmują bardziej bezpośrednie podejście i aproksymują funkcję zwracającą
najoptymalniejszą akcję dla danej obserwacji. W przeciwieństwie do Q-Learning agent nie musi wybrać akcji z największą
nagrodą, co czasem pozwala na mniej deterministyczne zachowanie pożądane w środowiskach losowych.

![img_5.png](img_5.png)

Zdecydowaliśmy, że sprawdzimy skuteczność najpopularniejszych algorytmów z obu tych kategorii, Deep Q Learning (DQN)
oraz Proximal Policy Optimization (PPO). Implementacje obu algorytmów udostępnia
biblioteka [OpenAI Baselines](https://github.com/openai/baselines).

### Architektura sieci neuronowej

Obie z metod uczenia wymagają zaprojektowania sieci neuronowej będącej w stanie przyjąć obserwacje jako wejście.
Jako że stan poszczególnych pól przedstawia dane przestrzenne, naturalne wydaje się wykorzystanie sieci
konwolucyjnych (CNN).
Popularnym podejściem w RL jest także łączenie sieci CNN (w charakterze "feature extractor") z dodatkowymi warstwami
MLP (typowa, w pełni połączona sieć neuronowa).
Zdecydowaliśmy, że przetestujemy różne architektury i rozmiary sieci.

### Testowane konfiguracje

#### PPO, CNN + 2xMLP(256)

Pierwsza testowana konfiguracja, używająca algorytmu PPO i korzystająca z połączenia CNN oraz z dwóch warstw MLP, każda
z 256 ukrytymi warstwami, radziła sobie słabo mimo dość długiej nauki.

- Ilość kroków: 7M
- Średnia długość gry: 60 kroków
- Średnia nagroda gry: -2

#### DQN, CNN + 2xMLP(64)

W następnej konfiguracji postanowiliśmy przetestować algorytm DQN w połączeniu z mniejszą siecią MLP. Wynikowa sieć dość
szybko osiągnęła wyniki lepsze niż poprzednia iteracja, ale potem zaczęła wykazywać dużą niestabilność i brak perspektyw
na dalszy wzrost.

- Ilość kroków: 700k
- Średnia długość gry: 65 kroków
- Średnia nagroda: -1,5

![img.png](img.png)

#### DQN, CNN + 2xMLP(256)

Postanowiliśmy zwiększyć rozmiar sieci MLP w nadziei na osiągnięcie dłuższego wzrostu, natomiast nie przyniosło to
zauważalnych skutków.

- Ilość kroków: 300k
- Średnia długość gry: 60 kroków
- Średnia nagroda: -2,5

![img_1.png](img_1.png)

#### DQN, 2xMLP(256)

W finalnej testowanej konfiguracji postanowiliśmy sprawdzić jakie efekty spowoduje wykorzystanie samej, dużej sieci MLP,
bez użycia CNN. Dzięki stosowanemu mechanizmowi treningu mieliśmy możliwość zmiany mechanizmu nagrody w trakcie nauki.
Postanowiliśmy więc stopniowo zwiększać karę za bezczynność (aż do -0.08) w momentach, w których agent przestawał
wykazywać wzrost uzyskiwanych nagród. Okazało się, że to podejście pozwoliło wytrenować obiektywnie najlepszego agenta.

- Ilość kroków: 20M
- Średnia długość gry: 75 kroków
- Średnia nagroda: -5

Niestety, przez zmieniony mechanizm nagrody ciężko porównywać powyższe liczby z poprzednimi wynikami. W następnej sekcji
dokonujemy bardziej obiektywnej ewaluacji.

![img_2.png](img_2.png)

### Ewaluacja

Ewaluacja modelu opierała się na wykonaniu 1000 gier. Wyniki:

- Średnia długość: 186 kroków
- Średnia liczba punktów zdobytych w grze: 15.2
- Średnia nagroda na każdy wykonany krok: 0,081

![](evaluate.gif)

### Wnioski

Wytrenowany agent osiąga wyniki porównywalne do ludzkich. Potrafi on planować proste strategie w celu uniknięcia
okrążenia, jednocześnie zdobywając rozsądną ilość punktów. Jednak czasem popełnia proste błędy, przykładowo wchodząc w
swój własny ogon przy bardziej skomplikowanych sytuacjach.

Ciekawym wynikiem jest uzyskana przy ewaluacji średnia nagroda za każdy wykonany krok, zbiegająca się z finalnym
stosunkiem kary za pasywność. Sugeruje to, że przy dalszym dostosowywaniu funkcji nagrody połączonej z dłuższym
treningiem być może udałoby się osiągnąć lepsze wyniki.

## Dokumentacja techniczna

### Instalacja

Do instalacji projektu wymagany jest Python 3.x. Aby zainstalować zależności, należy wykonać:

```shell
$ pip install -r requirements.txt
```

### Środowisko

Logikę gry realizuje implementacja środowiska zawarta w `env/snake_env.py`, z dodatkowymi narzędziami
w `env/snake_utils.py`. Dzięki modyfikacji funkcji step można modyfikować zasady przydzielania nagród:

```python
def step(self, action: Direction | None):
    if action is not None:
        self.state.direction = action
    done = self.__maybe_move_snake()
    if done:
        reward = -10
    else:
        if self.__maybe_consume_food():
            reward = 1
            done = len(self.state.body) > self.size.area * 0.5
        else:
            reward = -0.01
        self.__maybe_spawn_food()

    return self.state.build_tiles(), reward, done, {}
```

Działanie środowiska można przetestować za pomocą skryptu `test_human.py`, który pozwala kontrolować grę za pomocą
klawiatury.

```shell
$ python test_human.py
```

### Konfiguracja architektury sieci neuronowej

Architektura sieci jest definiowana w pliku `train.py`.

```python
model = DQN(
    policy=MlpPolicy,
    policy_kwargs={
        "net_arch": [256, 256],
    },
    # ...
)
```

Modyfikując te parametry, można zmieniać rozmiar sieci lub dodawać kolejne warstwy, których przykłady można znaleźć
w `snake_feature_extractor.py` lub `custom_combined_extractor.py`.

### Trenowanie

Aby rozpocząć trenowanie, należy uruchomić skrypt `train.py`. Jeśli w katalogu istnieje plik `model.zip`, jest on
wczytywany, a trenowanie jest wznawianie. W przeciwnym wypadku trening zaczyna się od nowa. Co 10k kroków skrypt
zapisuje aktualną wersję modelu, co później pozwala na kontynuację treningu.

```shell
$ python train.py
```

W trakcie treningu kluczowe metryki są logowane do katalogu `tb`. Można je monitorować korzystając z oprogramowania
TensorBoard.

```shell
$ tensorboard --logdir ./tb/
TensorBoard 2.11.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

Uruchomienie wskazanego adresu w przeglądarce otwiera konsolę, pokazującą aktualne parametry. Wśród nich za kluczowe
można uznać `ep_len_mean` (średnia długość gry) i `ep_rew_mean` (średnia nagroda w trakcie gry).

### Ewaluacja

Ewaluacja automatyczna jest możliwa za pomocą skryptu `evaluate_model.py`:

```shell
$ python evaluate_model.py best_model.zip
1000 episodes, avg len: 185.729, avg rew/episode 15.218, total rew/len 0.0819365850244173
```

Ręczne podejrzenie działania bota jest możliwe dzięki skryptowi `evaluate_model.py` który wyświetla graficzną
reprezentację działania agenta w prędkości czytelnej dla ludzkiego oka.

```shell
$ python test_model.py best_model.zip
```

## Bibliografia

1. [Faulty reward functions, OpenAI](https://openai.com/blog/faulty-reward-functions/)
2. [AlphaZero, Google Deep Mind](https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go)
3. [Snake with RL and CNN](https://www.researchgate.net/publication/327638529_Autonomous_Agents_in_Snake_Game_via_Deep_Reinforcement_Learning)
4. [SIMPLE framework](https://github.com/davidADSP/SIMPLE)
5. [Basic snake with RL](https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36)
6. [OpenAI Baselines](https://github.com/openai/baselines)