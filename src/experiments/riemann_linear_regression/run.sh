#!/bin/bash

for ID in {1..10}
do
    for alg in "IHT" "IHT-2"
    do
	python3 main.py $alg $ID
    done
done
