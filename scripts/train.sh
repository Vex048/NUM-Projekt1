#!/bin/bash
# =============== SKRYPT STARTOWY ===============
#
# Przykładowe użycie: 
#   ./scripts/train.sh
#   ./scripts/train.sh configs/inne_ustawienia.yaml
#
# ===============================================

# Eksportuj swój klucz do biblioteki Weights & Biases:
# Odkomentuj i wklej swój klucz jeśli nie był konfigurowany na poziomie globalnym.

CONFIG_PATH=${1:-configs/official_train.yaml}

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Błąd: Plik konfiguracji $CONFIG_PATH nie istnieje"
    exit 1
fi

echo "Start treningu z konfiguracją w: $CONFIG_PATH..."
python main.py --config $CONFIG_PATH
