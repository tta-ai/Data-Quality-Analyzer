#!/bin/bash

for i in "$@"
do
case $i in
    -i=*|--img=*)
    img="${i#*=}"
    ;;
    -m=*|--meta=*)
    meta="${i#*=}"
    ;;
    -d=*|--dataset=*)
    dataset="${i#*=}"
    ;;
    -p=*|--process=*)
    process="${i#*=}"
    ;;
    -c=*|--count=*)
    count="${i#*=}"
    ;;
    -n=*|--nworkers=*)
    nworkers="${i#*=}"
    ;;
    -v=*|--vector=*)
    vector="${i#*=}"
    ;;
    -re=*|--resize=*)
    resize="${i#*=}"
    ;;
    -ra=*|--ratio=*)
    ratio="${i#*=}"
    ;;
    -ms=*|--msample=*)
    msample="${i#*=}"
    ;;
    *)
            # unknown option
    ;;
esac
done

echo IMG = $img
echo META = $meta
echo DATASET = $dataset
echo PROCESS = $process
echo COUNT = $count
echo NWORKERS = $nworkers
echo VECTOR = $vector
echo RESIZE = $resize
echo RATIO = $ratio
echo MSAMPLE = $msample

for ((i=1;i<=$process;i++)); do
    python3 indicator.py --img $img --meta $meta --dataset $dataset --process $process --count $count --nworkers $nworkers --vector $vector --resize $resize --ratio $ratio --msample $msample &
done