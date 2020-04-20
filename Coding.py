# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


#DATASET CAN BE DOWNLOADED USING THE LINK https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/download

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shutil
import os
import imageio
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import scipy.ndimage as ndi

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os


#Storing the base directory path to the path_dir file
path_dir=os.getcwd()

#Code to check all the files present inside the input directory including files in sub folders too
for dirname, _, filenames in os.walk(path_dir+'/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.


import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras import backend as K
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


import os
print(os.listdir("../CORONA_HACKBOOK/input/Coronahack-Chest-XRay-Dataset/"))
print(os.listdir("../CORONA_HACKBOOK/input/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"))



# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)




os.mkdir(path_dir+"/corona_check")
os.mkdir(path_dir+"/corona_check/train")
os.mkdir(path_dir+"/corona_check/test")
os.mkdir(path_dir+"/corona_check/train/Normal/")
os.mkdir(path_dir+"/corona_check/train/COVID19/")
os.mkdir(path_dir+"/corona_check/test/Normal/")
os.mkdir(path_dir+"/corona_check/test/COVID19/")


def copy_img(src_path, dst_path):
    try:
        shutil.copy(src_path, dst_path)
        stmt = 'File Copied'
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
        stmt = 'Copy Failed - IO Error'
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))
        stmt = 'Copy Failed - other Error' + sys.exc_info()

    return stmt


data_dir="/home/arnab/PycharmProjects/CORONA_HACKBOOK/input/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"
metadata_path="/home/arnab/PycharmProjects/CORONA_HACKBOOK/input/Coronahack-Chest-XRay-Dataset/"


train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')



meta_data = pd.read_csv(metadata_path+'Chest_xray_Corona_Metadata.csv')
print(meta_data.head())


meta_data['File_path']=''
meta_data.loc[meta_data['Dataset_type']=='TRAIN','File_path']=train_dir+'/'
meta_data.loc[meta_data['Dataset_type']=='TEST','File_path']=test_dir+'/'

meta_data['X_ray_img_nm_path']=meta_data['File_path']+meta_data['X_ray_image_name']

print(meta_data.head())




meta_COVID_19_train = meta_data[(meta_data['Dataset_type']=='TRAIN') &
                        ((meta_data['Label']=='Normal')|(meta_data['Label']=='Pnemonia') & (meta_data['Label_2_Virus_category']=='COVID-19'))]


meta_COVID_19_test = meta_data[(meta_data['Dataset_type']=='TEST') &
                        ((meta_data['Label']=='Normal')|(meta_data['Label']=='Pnemonia') & (meta_data['Label_2_Virus_category']=='COVID-19'))]


## Moving the 10 Corona Infected dataset to Test

meta_data_covid_test = meta_data[meta_data['Label_2_Virus_category']=='COVID-19'].sample(12)
meta_COVID_19_train = meta_COVID_19_train[~meta_COVID_19_train['X_ray_image_name'].isin(meta_data_covid_test['X_ray_image_name'])]
meta_COVID_19_test_fnl = pd.concat([meta_data_covid_test,meta_COVID_19_test],ignore_index=False)



meta_COVID_19_train.loc[meta_COVID_19_train['Label'] =='Pnemonia','Label']='COVID19'
meta_COVID_19_test_fnl.loc[meta_COVID_19_test_fnl['Label'] =='Pnemonia','Label']='COVID19'



print("===============Train Set==========================\n")
print(meta_COVID_19_train.groupby(['Label']).agg({'Dataset_type':'count'}))

print("\n===============Test Set==========================\n")
print(meta_COVID_19_test_fnl.groupby(['Label']).agg({'Dataset_type':'count'}))


meta_COVID_19_train['Img_tgt_path']="/home/arnab/PycharmProjects/CORONA_HACKBOOK/input/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/"
meta_COVID_19_test_fnl['Img_tgt_path']="/home/arnab/PycharmProjects/CORONA_HACKBOOK/input/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"


meta_COVID_19_train['Move_status'] = np.vectorize(copy_img)(meta_COVID_19_train['X_ray_img_nm_path'],meta_COVID_19_train['Img_tgt_path'])
meta_COVID_19_test_fnl['Move_status'] = np.vectorize(copy_img)(meta_COVID_19_test_fnl['X_ray_img_nm_path'],meta_COVID_19_test_fnl['Img_tgt_path'])


















































meta_COVID_19_train.loc[meta_COVID_19_train['Label']=='Normal','Img_tgt_path']=meta_COVID_19_train['Img_tgt_path']+'Normal/'
meta_COVID_19_train.loc[meta_COVID_19_train['Label']=='COVID19','Img_tgt_path']=meta_COVID_19_train['Img_tgt_path']+'COVID19/'

meta_COVID_19_test_fnl.loc[meta_COVID_19_test_fnl['Label']=='Normal','Img_tgt_path']=meta_COVID_19_test_fnl['Img_tgt_path']+'Normal/'
meta_COVID_19_test_fnl.loc[meta_COVID_19_test_fnl['Label']=='COVID19','Img_tgt_path']=meta_COVID_19_test_fnl['Img_tgt_path']+'COVID19/'