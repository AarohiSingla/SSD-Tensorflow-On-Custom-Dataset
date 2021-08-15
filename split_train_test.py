import glob
import os
import numpy as np
import sys

current_dir = "VOC2007/JPEGImages"
split_pct = 10;
file_train = open("VOC2007/ImageSets/train.txt", "w")  
file_val = open("VOC2007/ImageSets/test.txt", "w")  
counter = 1  
index_test = round(100 / split_pct)  
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.png")):  
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if counter == index_test:
                counter = 1
                file_val.write(current_dir + "/" + title + '.png' + "\n")
        else:
                file_train.write(current_dir + "/" + title + '.png' + "\n")
                counter = counter + 1
file_train.close()
file_val.close()