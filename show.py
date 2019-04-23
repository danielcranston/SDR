import matplotlib.pyplot as plt
import sys
import numpy as np
import imageio

sys.path.append('/home/dcranston/Documents/nconv')
from utils.python_pfm import readPFM, writePFM


# a = imageio.imread('data/Test/LmidL_denseGT.png').astype('float32')
# a = -a / 64 + 350
# writePFM('data/Test/disp0GT.pfm', a.astype('float32'))

if False:
	gt, _      = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/MiddV3/trainingH/ArtL/disp0GT.pfm')
	init, _    = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/MiddV3/trainingH/ArtL/disp_WTA.pfm')
	rgb = imageio.imread('/home/dcranston/Documents/Exjobb/SDR/data/MiddV3/trainingH/ArtL/im0.png')
	output, _  = readPFM('/home/dcranston/Documents/Exjobb/SDR/results/trainingH/ArtL/disp0FDR.pfm')
else:
	gt, _      = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/Test/disp0GT.pfm')
	init, _    = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/Test/disp_WTA.pfm')
	rgb = imageio.imread('/home/dcranston/Documents/Exjobb/SDR/data/Test/im0.png')
	output, _  = readPFM('/home/dcranston/Documents/Exjobb/SDR/results/Test/disp0FDR.pfm')

maxVal = max(init.max(), output.max())
minVal = min(init.min(), output.min())



plt.figure('init'); plt.imshow(init, vmin=minVal, vmax=maxVal)
plt.figure('gt'); plt.imshow(gt, vmin=minVal, vmax=maxVal)
plt.figure('output'); plt.imshow(output, vmin=minVal, vmax=maxVal)
plt.figure('rgb'); plt.imshow(rgb)
plt.show()