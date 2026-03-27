### 1. Stworzyć baseline na samych obrazach - Jakiś XGBoost albo prosty własny CNN

### 2. Prztestsotwac ResNet18 z oraz bez pretrained_weights

### 3. Użyć innej funckji celu może focalCrossEntropy - bo jest imbalancja klas

Albo zrobic coś na datasetcie

### 4. Spróbować podejście obrazki + metadane

Czyli ResNet, któremu odcinamy ostatni layer klasyfikujący, żeby zwracał wektor embeddingów
I tworzymy drugą sieć, która będzie działała na metadanych oraz tych embeddingach do klasyfikacji
Trzeba będzie poczyścić metadane, bo mogą być jakeiś nulle i zroobić feautre enginnering

### 5. Odpalić optunę na podejściu, które będzie dawało najlepsze początkowe wyniki

### 6. Puścić ostateczny trening na hiperparametrach odplanych przez optune
