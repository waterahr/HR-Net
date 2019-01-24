# HR-Net
Our proposed model considers the abstraction level of the layer feature map with the attribute semantics together, and utilize the high-level attribute reasoning based on their dependencies on the low-level attributes. We show in an extensive evaluation on the PETA dataset that our proposed hierarchical reason attribute recognition model provides competitive performance for each attribute,  including the low-level attributes and high-level attributes, and improves on the published state-of-the-art on the dataset. The paper is published at [CVPR'19](http://arxiv.org/pdf/1512.04150.pdf).

The framework of the HR-Net is as below:
![Framework](https://github.com/waterahr/HR-Net/images/framework.jpg)

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
* Install [caffe](https://github.com/BVLC/caffe), compile the matcaffe (matlab wrapper for caffe), and make sure you could run the prediction example code classification.m.
* Clone the code from Github:
```
git clone https://github.com/metalbubble/CAM.git
cd CAM
```
* Download the pretrained network
```
sh models/download.sh
```
* Run the demo code to generate the heatmap: in matlab terminal, 
```
demo
```
* Run the demo code to generate bounding boxes from the heatmap: in matlab terminal,
```
generate_bbox
```

The demo video of what the CNN is looking is [here](https://www.youtube.com/watch?v=fZvOy0VXWAI). The reimplementation in tensorflow is [here](https://github.com/jazzsaxmafia/Weakly_detector). The pycaffe wrapper of CAM is reimplemented at [here](https://github.com/gcucurull/CAM-Python).

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
