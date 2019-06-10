import numpy as np
import sys
import re
import imageio
import matplotlib.pyplot as plt
import matplotlib
from eval_utils.io import readPFM, writePFM, scale_disp
from eval_utils.metrics import create_category_mask


pair = sys.argv[1] + '_pair'
item = sys.argv[2].zfill(3)
root = 'data/liu_dataset/' 

epethresh = 4

wta, scale = readPFM(root + pair + '/left_initial_disparity/' + item + '.pfm')
# output, scale = readPFM(root + pair + '/left_output_sdr/' + item + '.pfm')
output = scale_disp(imageio.imread(root + pair + '/left_inpaint_disparity/' + item + '.png'))
try:
	gt = imageio.imread(root + pair + '/left_gt_merged/' + item + '.png')
	print('loaded merged GT')
except:
	gt = imageio.imread(root + pair + '/left_gt/' + item + '.png')
	
rgb = imageio.imread(root + pair + '/left_rgb/' + item + '.png')

gt = np.abs(scale_disp(gt))

comb, cmap = create_category_mask(wta, gt)
valids = gt != 0
epe_out = valids * (np.abs(output - gt))

epe1 = epe_out * (comb == 1)
epe2 = epe_out * (comb == 2)
epe3 = epe_out * (comb == 3)
epe1[np.invert(comb == 1)] = np.inf
epe2[np.invert(comb == 2)] = np.inf
epe3[np.invert(comb == 3)] = np.inf
cmap_bad = matplotlib.cm.viridis
cmap_bad.set_bad('gray', 1.)
epe_out[np.invert(valids)] = np.inf

maxVal = max(wta.max(), gt.max(), output.max())
minVal = min(wta.min(), gt.min(), output.min())

plt.figure('wta'); plt.imshow(wta, vmin=minVal, vmax=maxVal); plt.axis('off')
plt.figure('output'); plt.imshow(output, vmin=minVal, vmax=maxVal); plt.axis('off')
plt.figure('gt'); plt.imshow(gt, vmin=minVal, vmax=maxVal); plt.axis('off')
plt.figure('rgb'); plt.imshow(rgb); plt.axis('off')
# plt.figure('epe_wta'); plt.imshow(epe_wta, vmin=0, vmax=20)
plt.figure('all_epe{}'.format(epethresh)); plt.imshow(epe_out, cmap=cmap_bad, vmin=0, vmax=epethresh); plt.axis('off')
plt.figure('cat1_epe{}'.format(epethresh)); plt.imshow(epe1, cmap=cmap_bad, vmin=0, vmax=epethresh); plt.axis('off')
plt.figure('cat2_epe{}'.format(epethresh)); plt.imshow(epe2, cmap=cmap_bad, vmin=0, vmax=epethresh); plt.axis('off')
plt.figure('cat3_epe{}'.format(epethresh)); plt.imshow(epe3, cmap=cmap_bad, vmin=0, vmax=epethresh); plt.axis('off')
plt.figure('comb'); plt.imshow(comb, cmap=cmap); plt.axis('off')
plt.show()