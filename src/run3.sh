#!/bin/bash
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -g 1 -b 128 -i 50 -s 0
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -w 320x120binary51_shuffleadam_newnewhier_oldreason_newlossnoexp8multi_split0_iter50 -g 1 -s 0
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -g 1 -b 128 -i 50 -s 1
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -w 320x120binary51_shuffleadam_newnewhier_oldreason_newlossnoexp8multi_split1_iter50 -g 1 -s 1
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -g 1 -b 128 -i 50 -s 2
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -w 320x120binary51_shuffleadam_newnewhier_oldreason_newlossnoexp8multi_split2_iter50 -g 1 -s 2
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -g 1 -b 128 -i 50 -s 3
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -w 320x120binary51_shuffleadam_newnewhier_oldreason_newlossnoexp8multi_split3_iter50 -g 1 -s 3
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -g 1 -b 128 -i 50 -s 4
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -w 320x120binary51_shuffleadam_newnewhier_oldreason_newlossnoexp8multi_split4_iter50 -g 1 -s 4

is_train=$1
gpus=$2
for spl in 0 1 2 3 4
do
    if [ $is_train -eq 1 ]
    then
        echo "python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -g $2 -b 128 -i 50 -s $spl"
        python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -g $2 -b 128 -i 50 -s $spl
    else
        echo "python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -w 320x120binary51_shuffleadam_newnewhier_oldreason_focalloss8multi2gamma_split${spl}_iter50 -g $2 -s $spl"
        python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v3 -c 51 -w 320x120binary51_shuffleadam_newnewhier_oldreason_focalloss8multi2gamma_split${spl}_iter50 -g $2 -s $spl
    fi
done