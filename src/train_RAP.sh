python train_RAP.py -g 0 -c 51 -m GoogLeNet -b 64 -w ../models/imagenet_models/GoogLeNet_RAP/binary51_final_model.h5
python train_RAP_hiarchical.py -m hiarBayesGoogLeNet -b 64 -g 0 -c 51 -w ../models/imagenet_models/hiarBayesGoogLeNet_RAP/binary51_final_model.h5
python train_RAP_hiarchical.py -m hiarGoogLeNet -b 64 -g 0 -c 51 -w ../models/imagenet_models/hiarGoogLeNet_RAP/binary51_final_model.h5