import glob
import pandas as pd
import numpy as np
import scipy.io

dir_path = glob.glob('./part-affordance-dataset/tools/*')


object_to_cls ={
                'bowl': 0,
                'cup': 1,
                'hammer': 2,
                'knife': 3, 
                'ladle': 4,
                'mallet': 5,
                'mug': 6,
                'pot': 7,
                'saw': 8,
                'scissors': 9,
                'scoop': 10,
                'shears': 11,
                'shovel': 12,
                'spoon': 13,
                'tenderizer': 14,
                'trowel': 15,
                'turner':16
}


image_path = []
image_level_aff_path = []
obj_list = []
pixel_level_aff_path = []


for d in dir_path:
    img_path = glob.glob(d + '/*.jpg')
    o = [object_to_cls[d[32:-3]]]    # path[32:-3] => object name
    
    for img in img_path:
        multi_hot = np.zeros(7, dtype=np.int64)
        pix_lev_aff_path = img[:-7] + 'label.mat'
        label = scipy.io.loadmat(pix_lev_aff_path)['gt_label']
        for i in range(1, 8):
            if i in label:
                multi_hot[i-1] = 1
        
        image_path.append(img)
        image_level_aff_path.append(img[:-7] + 'label.npy')
        obj_list += o
        pixel_level_aff_path.append(pix_lev_aff_path)
        np.save(img[:-7] + 'label.npy', multi_hot)


image_train = []
image_test = []
image_level_aff_train = []
image_level_aff_test = []
obj_train = []
obj_test = []
pixel_level_aff_train = []
pixel_level_aff_test = []

for i, (img, img_lev_aff, obj, pix_lev_aff) in enumerate(zip(image_path, image_level_aff_path, obj_list, pixel_level_aff_path)):
    if i%5 == 0:
        image_test.append(img)
        image_level_aff_test.append(img_lev_aff)
        obj_test.append(obj)
        pixel_level_aff_test.append(pix_lev_aff)
    else:
        image_train.append(img)
        image_level_aff_train.append(img_lev_aff)
        obj_train.append(obj)
        pixel_level_aff_train.append(pix_lev_aff)


df_train = pd.DataFrame({
    'image': image_train,
    'image_level_affordance': image_level_aff_train,
    'object': obj_train,
    'pixel_level_affordance': pixel_level_aff_train},
    columns=['image', 'image_level_affordance', 'object', 'pixel_level_affordance']
)


df_test = pd.DataFrame({
    'image': image_test,
    'image_level_affordance': image_level_aff_test,
    'object': obj_test,
    'pixel_level_affordance': pixel_level_aff_test},
    columns=['image', 'image_level_affordance', 'object', 'pixel_level_affordance']
)


data = pd.concat([df_train, df_test])


df_train.to_csv('./part-affordance-dataset/train.csv', index=None)
df_test.to_csv('./part-affordance-dataset/test.csv', index=None)
data.to_csv('./part-affordance-dataset/all_data.csv', index=None)
