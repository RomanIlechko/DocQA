#!/bin/bash

# Get the directory of the input file
DIR="$(dirname "$1")"

echo "FROM $1 $PROMT" | cat > "$DIR/Modelfile"
# Create the model using ollama
echo "$2-$3"
ollama create "$2-$3" -f "$DIR/Modelfile"
