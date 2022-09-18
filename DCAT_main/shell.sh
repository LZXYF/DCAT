#!/bin/bash

for x in 1 2 3 4 5 6
do
    source activate emlm
    nohup python DCAT_main.py --gcn_layers=$x &
    wait
done
