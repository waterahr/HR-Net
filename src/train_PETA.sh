#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNetSPP
#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNet
python train_PETA_hiarchical.py -m hiarGoogLeNetSPP -c 61 -g 1
python train_PETA_hiarchical.py -g 1 -m hiarGoogLeNet -c 61 
python train_PETA_hiarchical.py -g 1 -m hiarBayesGoogLeNet -c 61
python train_PETA_hiarchical.py -g 1 -m hiarGoogLeNetWAM -c 61
python train_PETA_hiarchical.py -g 1 -m hiarGoogLeNet_mid -c 61
python train_PETA_hiarchical.py -g 1 -m hiarGoogLeNet_low -c 61
#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNet -d 1 -w ../models/imagenet_models/GoogLeNet_PETA/binary61_epoch50_valloss0.15.hdf5
#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNet -d 2 -w ../models/imagenet_models/GoogLeNet_PETA/binary61_epoch50_valloss0.15.hdf5
#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNet -d 3 -w ../models/imagenet_models/GoogLeNet_PETA/binary61_epoch50_valloss0.15.hdf5
#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNet -d 4 -w ../models/imagenet_models/GoogLeNet_PETA/binary61_epoch50_valloss0.15.hdf5
#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNet -d 5 -w ../models/imagenet_models/GoogLeNet_PETA/binary61_epoch50_valloss0.15.hdf5
#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNet -d 6 -w ../models/imagenet_models/GoogLeNet_PETA/binary61_epoch50_valloss0.15.hdf5
#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNet -d 7 -w ../models/imagenet_models/GoogLeNet_PETA/binary61_epoch50_valloss0.15.hdf5
#python train_PETA.py -g 1 -c 61 -b 256 -m GoogLeNet -d 8 -w ../models/imagenet_models/GoogLeNet_PETA/binary61_epoch50_valloss0.15.hdf5