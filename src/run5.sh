#!/bin/bash
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -b 64 -g 2 -s 0 -i 50
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -w 320x120binary51_shuffleadam_newnewnewnewhier_oldreason_newlossnoexp5multi_split0_iter50 -g 2 -s 0
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -b 64 -g 2 -s 1 -i 50
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -w 320x120binary51_shuffleadam_newnewnewnewhier_oldreason_newlossnoexp5multi_split1_iter50 -g 2 -s 1
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -b 64 -g 2 -s 2 -i 50
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -w 320x120binary51_shuffleadam_newnewnewnewhier_oldreason_newlossnoexp5multi_split2_iter50 -g 2 -s 2
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -b 64 -g 2 -s 3 -i 50
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -w 320x120binary51_shuffleadam_newnewnewnewhier_oldreason_newlossnoexp5multi_split3_iter50 -g 2 -s 3
#python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -b 64 -g 2 -s 4 -i 50
#python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -w 320x120binary51_shuffleadam_newnewnewnewhier_oldreason_newlossnoexp5multi_split4_iter50 -g 2 -s 4


is_train=$1
gpus=$2
for spl in 0 1 2 3 4
do
    if [ $is_train -eq 1 ]
    then
        echo "python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -g $2 -b 128 -i 50 -s $spl"
        python train_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -g $2 -b 128 -i 50 -s $spl
    else
        echo "python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -w 320x120binary51_shuffleadam_newnewnewnewhier_oldreason_focalloss6multi2gamma_split${spl}_iter50 -g $2 -s $spl"
        python test_RAP_hiarchical.py -m hiarBayesGoogLeNet_gap_v5 -c 51 -w 320x120binary51_shuffleadam_newnewnewnewhier_oldreason_focalloss6multi2gamma_split${spl}_iter50 -g $2 -s $spl
    fi
done