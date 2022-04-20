#!/bin/bash

# input & output
if [ "$#" -lt 1 ]; then
    echo "Input file missing"
    exit 1
else
    IN_FILE="$1"
fi

if [ "$#" -lt 2 ]; then
    OUT_FILE="out.h5"
else
    OUT_FILE="$2"
fi

client.py -a 127.0.0.1 -G images -c bart_pulseq -o $OUT_FILE $IN_FILE
