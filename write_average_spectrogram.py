"""
Write the average warped spectrogram.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image
from scipy.ndimage import zoom


DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/juvi_1/'
NB_FN = os.path.join(DATA_DIR, 'nb_20180305_spec_calcium.npy')

DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/juvi_1/'
NB_FN = os.path.join(DATA_DIR, 'nb_20180228_spec_calcium.npy')

DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk215/'
NB_FN = os.path.join(DATA_DIR, 'nb_10052018_spec_calcium.npy')

DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk215/'
NB_FN = os.path.join(DATA_DIR, 'nb_09292018_spec_calcium.npy')

# DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blk202/'
# NB_FN = os.path.join(DATA_DIR, 'nb_spec_calcium.npy')
#
# DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/org137/'
# NB_FN = os.path.join(DATA_DIR, 'nb_spec_calcium.npy')
#
# DATA_DIR = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/yel141/'
# NB_FN = os.path.join(DATA_DIR, 'nb_spec_calcium.npy')


if __name__ == '__main__':
	mag_factor = 3
	d = np.load(NB_FN, allow_pickle=True).item()
	spec = np.mean(d['spec'], axis=0).reshape(128,128)[::-1]
	rgbArray = np.zeros((128,128,3), dtype='int')
	temp_max = np.quantile(np.abs(spec), 0.9975)
	print("max diff:", temp_max)
	spec /= temp_max
	# # Green/magenta.
	rgbArray[..., 0] = spec * 255 # R
	rgbArray[..., 1] = spec * 255 # G
	rgbArray[..., 2] = spec * 255 # B
	rgbArray[rgbArray<0] = 0
	rgbArray[rgbArray>255] = 255
	temp_rgbArray = np.zeros((128*mag_factor,128*mag_factor,3))
	for i in range(3):
		temp_rgbArray[:,:,i] = zoom(rgbArray[:,:,i].astype(np.float), mag_factor, order=3)
	rgbArray = temp_rgbArray.clip(0,255).astype(np.int)
	img = Image.fromarray(np.array(rgbArray, 'uint8'))
	img.save('temp.jpeg')

###
