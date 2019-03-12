#!/bin/bash

<<COMMENT
for i in {2..4}
do
    #echo $i
    python train_RAP_hiarchical.py -m hiarBayesGoogLeNet -c 51 -b 64 -g 0 -s $i -wd 75 -hg 160 -i 200
done
COMMENT

#<<COMMENT
split=(0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4)
idx=0
for weight in $(cat ./weights.txt)
do
    #echo $i
    echo $weight
    echo ${split[$idx]}
    python test_RAP_hiarchical.py -m hiarGoogLeNet -c 51 -g 0 -wd 227 -hg 227 -w $weight #-s ${split[$idx]}
    idx=$[$idx+1]
done
#COMMENT