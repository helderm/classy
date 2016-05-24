import os
import numpy as np
from PIL import Image
import shutil

def subdata():
    base_data_path = './data/images/'
    mark = 0
    count = np.zeros(4)

    # get all images paths
    for brand in os.listdir(base_data_path):
        brand_path = os.path.join(base_data_path, brand)
        if os.path.isdir(brand_path):
            for fashionshow in os.listdir(brand_path):
                #get labels
                    #0: Spring
                    #1: Fall
                    #2: Resort
                    #3: Pre-fall
                image_label = -1
                if fashionshow[0] == 'S':
                    image_label = 0
                elif fashionshow[0] == 'F':
                    image_label = 1
                elif fashionshow[0] == '2':
                    if fashionshow[4] == 'R':
                        image_label = 2
                    if fashionshow[4] == 'P':
                        image_label = 3
                if (image_label == -1):
                    print('Label could not be identified')

                fashionshow_path = os.path.join(brand_path, fashionshow)
                if os.path.isdir(fashionshow_path):
                    for image in os.listdir(fashionshow_path):
                        image_path = os.path.join(fashionshow_path, image)
                        new_path = os.path.join('./data/', str(image_label))
                        if not os.path.exists(new_path):
                            os.makedirs(new_path)

                        shutil.copy(image_path, new_path)

                        dst_file = os.path.join(new_path, image)
                        new_name = 'img_'+str(int(count[image_label]))+'.jpg'
                        new_file = os.path.join(new_path, new_name)
                        os.rename(dst_file, new_file)
                        count[image_label] += 1
        mark += 1
        print('Finish ',mark, '\t', brand)


    print(count)

if __name__ == '__main__':
    subdata()