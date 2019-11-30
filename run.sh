#!/bin/bash

img=$1
meta=$2
dataset=$3
process=$4
count=$5
nworkers=$6
vector=$7
resize=$8
ratio=$9
msample=${10}

for ((i=1;i<=$process;i++)); do
    python3 indicator.py --img $img --meta $meta --dataset $dataset --process $process --count $count --nworkers $nworkers --vector $vector --resize $resize --ratio $ratio --msample $msample &
done