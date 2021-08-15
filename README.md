# SSD-Tensorflow-On-Custom-Dataset

Single Shot Detector on Custom dataset. 
I am working on 2 classes : 1st is headphone and 2nd class is earphone.

Clone this repo and do few modifications and your Custom Object Detector using SSD will be ready.

google_colab_example.ipynb will show you how to train SSD on Google Colab.

To understand the whole functionality, Check this video:  https://youtu.be/HYWS4jh0i4Y

## To train your model: Use this command:
python train_ssd_network.py --dataset_name=pascalvoc_2007 --dataset_split_name=train --model_name=ssd_300_vgg --save_summaries_secs=60 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.001 --batch_size=6 --gpu_memory_fraction=0.9 --checkpoint_exclude_scopes =ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box

