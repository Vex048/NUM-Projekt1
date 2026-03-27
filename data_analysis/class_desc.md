### Nowotwory złośliwe (Krytyczne do wykrycia)

- Czerniak (mel - Melanoma) Najbardziej niebezpieczny i agresywny rak skóry ze wszystkich tu obecnych. Daje przerzuty. Wizualnie często charakteryzuje się asymetrią, nieregularnymi brzegami i wielobarwnością (czarny, brązowy, niebieski, biały). Naszym ostatecznym celem biznesowym/medycznym jest to, aby model miał na tej klasie maksymalny Recall (czułość) – nie możemy sobie pozwolić na fałszywie negatywne wyniki (False Negatives).

- Rak podstawnokomórkowy (bcc - Basal cell carcinoma)
  Najczęstszy rodzaj raka skóry. Rzadko daje przerzuty, ale jest złośliwy miejscowo (potrafi niszczyć okoliczne tkanki). W dermatoskopii często widać na nim charakterystyczne, rozgałęziające się naczynia krwionośne (tzw. "drzewkowate").

### Zmiany przednowotworowe (Wysokie ryzyko)

- Rogowacenie słoneczne / Choroba Bowena (akiec - Actinic keratoses and intraepithelial carcinoma)
  Są to zmiany wywołane przewlekłym uszkodzeniem przez słońce (UV). Traktuje się je jako wczesne, nieinwazyjne stadium raka (rak wewnątrznabłonkowy). Często są czerwone, łuszczące się i szorstkie. Z perspektywy modelu, to klasa, którą często myli się z innymi zmianami zapalnymi.

### Zmiany łagodne (Niegroźne)

- Znamiona melanocytowe (nv - Melanocytic nevi)
  To są zwykłe "pieprzyki". Są łagodne i całkowicie bezpieczne. To właśnie ta klasa, która stanowi około 67% całego naszego zbioru HAM10000. Przez tę gigantyczną nadreprezentację nasz model na początku treningu będzie próbował zgadywać nv dla każdego zdjęcia, żeby sztucznie zawyżyć sobie ogólne Accuracy (dlatego samo Accuracy to tutaj śmieciowa metryka).

- Łagodne zmiany rogowe (bkl - Benign keratosis-like lesions)
  Kategoria obejmująca m.in. brodawki łojotokowe czy plamy soczewicowate. Zmiany całkowicie łagodne, uwarunkowane genetycznie lub wiekiem. Problem polega na tym, że dla niewprawnego oka (i dla słabej sieci konwolucyjnej) mogą wizualnie bardzo przypominać czerniaka (mel), co czyni je trudnymi przypadkami brzegowymi (edge cases).

- Włókniak skóry (df - Dermatofibroma)
  Niegroźna zmiana, która często powstaje w wyniku drobnego urazu (np. po ukąszeniu owada czy zadrapaniu). To małe, twarde guzki. W HAM10000 jest ich bardzo mało (najmniej ze wszystkich klas, często nieco ponad 100 zdjęć), co czyni z nich kolejną trudność przy balansowaniu zbioru.

- Zmiany naczyniowe (vasc - Vascular lesions)
  Różnego rodzaju naczyniaki. Nie powstają z komórek barwnikowych (jak pieprzyki), ale z drobnych naczyń krwionośnych. Przez to na zdjęciach często mają bardzo specyficzny czerwony, fioletowy lub niebiesko-czarny kolor. Dla modelu zazwyczaj dość łatwe do wyłapania ze względu na unikalną paletę barw.
