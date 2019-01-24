# HR-Net
Our proposed model considers the abstraction level of the layer feature map with the attribute semantics together, and utilize the high-level attribute reasoning based on their dependencies on the low-level attributes. We show in an extensive evaluation on the PETA dataset that our proposed hierarchical reason attribute recognition model provides competitive performance for each attribute,  including the low-level attributes and high-level attributes, and improves on the published state-of-the-art on the dataset. The paper is published at [CVPR'19](http://arxiv.org/pdf/1512.04150.pdf).

The framework of the HR-Net is as below:  
![Framework](https://github.com/waterahr/HR-Net/tree/master/images/framework.png)

### NEW: Tensorflow code
You also could take a look at the [HR-Net_tensorflow](https://github.com/waterahr/HR-Net_tensorflow) to see the tensorflow version.

### Pre-trained models in Caffe:
* GoogLeNet-CAM model on ImageNet: ```models/deploy_googlenetCAM.prototxt``` weights:  [http://cnnlocalization.csail.mit.edu/demoCAM/models/imagenet_googlenetCAM_train_iter_120000.caffemodel]
* VGG16-CAM model on ImageNet: ```models/deploy_vgg16CAM.prototxt``` weights:  [http://cnnlocalization.csail.mit.edu/demoCAM/models/vgg16CAM_train_iter_90000.caffemodel]
* GoogLeNet-CAM model on Places205: ```models/deploy_googlenetCAM_places205.prototxt``` weights:  [http://cnnlocalization.csail.mit.edu/demoCAM/models/places_googlenetCAM_train_iter_120000.caffemodel]
* AlexNet+-CAM on ImageNet:```models/deploy_alexnetplusCAM_imagenet.prototxt``` weights:  [http://cnnlocalization.csail.mit.edu/demoCAM/models/alexnetplusCAM_imagenet.caffemodel]
* AlexNet+-CAM on Places205 (used in the [online demo](http://places.csail.mit.edu/demo.html)):  ```models/deploy_alexnetplusCAM_places205.prototxt``` weights:[http://cnnlocalization.csail.mit.edu/demoCAM/models/alexnetplusCAM_places205.caffemodel]

### Pre-trained GoogLeNet model weights:
GoogLeNet model on ImageNet: ```./results/googlenet_weights.npy``` weights:  
[https://drive.google.com/open?id=1MMgfdNcO7uoNtQM8Tarsa8etyCbwbVd-]

### Usage Instructions:
* Clone the code from Github:
```
git clone https://github.com/waterahr/HR-Net.git
cd HR-Net
```
* Download the pretrained network
```
wget https://drive.google.com/open?id=1MMgfdNcO7uoNtQM8Tarsa8etyCbwbVd-
```
* Download the [PETA](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html) dataset
* Run the code ```./notebook/generating_pandas_dataframe_file.ipynb``` to generate the csv file to train and test the model
* Train the model on the training set of PETA, the model file will be stored at ```./models/imagenet_models/MODELNAME_PETA```
```
python train_PETA_hiarchical.py -g GPUSINDEX -m MODELNAME -c CLASSNUM -w WEIGHTSFILEPATH -wd IMAGEWIDTH -hg IMAGEHEIGHT -i INITIALEPOCH -b BATCHSIZE
```
* Test the model on the testing dataset of PETA, the prediction file will be stored at ```./results/predictions/MODELNAME_SAVENAME_WEIGHTSNAME_predictions_imagenet_test7600.npy```
```
python test_PETA_hiarchical.py -g GPUSINDEX -m MODELNAME -c CLASSNUM -w WEIGHTSFILEPATH -wd IMAGEWIDTH -hg IMAGEHEIGHT
```
* Run the web demo to show the pedestrian image and its attribute recognition result
```
python app.py -g GPUSINDEX
```
Then open your browser and enter the URL：```http://[your IP address running the app.py]:[your port running the app.py]```

### Reference:
```
@inproceedings{zhou2016cvpr,
    author    = {Zhou, Bolei and Khosla, Aditya and Lapedriza, Agata and Oliva, Aude and Torralba, Antonio},
    title     = {Learning Deep Features for Discriminative Localization},
    booktitle = {Computer Vision and Pattern Recognition},
    year      = {2016}
}
```

### License:
The pre-trained models and the HR-Net technique are released for unrestricted use.

Contact [H.R. An](waterahr@gmail.com) if you have questions.
