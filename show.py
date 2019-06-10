import matplotlib.pyplot as plt
import sys
import numpy as np
import imageio

from eval_utils.io import readPFM, writePFM, scale_disp
from eval_utils.visualization import show_overview, show_plots
from eval_utils.metrics import create_category_mask


if len(sys.argv) != 2: raise IOError('specify midd test or liu')
mode = sys.argv[1]

plot = 2
epethresh = 20

if mode == 'midd':
	gt, _      = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/MiddV3/trainingH/ArtL/disp0GT.pfm')
	init, _    = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/MiddV3/trainingH/ArtL/disp_WTA.pfm')
	rgb = imageio.imread('/home/dcranston/Documents/Exjobb/SDR/data/MiddV3/trainingH/ArtL/im0.png')
	output, _  = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/MiddV3/trainingH/ArtL/disp0FDR.pfm')
elif mode == 'test':
	gt, _      = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/Test/disp0GT.pfm')
	init, _    = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/Test/disp_WTA.pfm')
	rgb = imageio.imread('/home/dcranston/Documents/Exjobb/SDR/data/Test/im0.png')
	output, _  = readPFM('/home/dcranston/Documents/Exjobb/SDR/results/Test/disp0FDR.pfm')
elif mode == 'liu':
	gt, _      = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/liu_dataset/left_pair/left_gt/021.pfm')
	init, _    = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/liu_dataset/left_pair/left_initial_disparity/021.pfm')
	rgb = imageio.imread('/home/dcranston/Documents/Exjobb/SDR/data/liu_dataset/left_pair/left_rgb/021.png')
	output, _  = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/liu_dataset/left_pair/left_output_sdr/021.pfm')
elif mode == 'liu_sparser':
	gt, _      = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/liu_dataset/left_pair/left_gt/022.pfm')
	init, _    = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/liu_dataset/left_pair/left_initial_disparity_sparser/022.pfm')
	rgb = imageio.imread('/home/dcranston/Documents/Exjobb/SDR/data/liu_dataset/left_pair/left_rgb/022.png')
	output, _  = readPFM('/home/dcranston/Documents/Exjobb/SDR/data/liu_dataset/left_pair/left_output_sdr_sparser/022.pfm')
else:
	raise IOError('specify midd test or liu')
epe = np.abs(output - gt) * (gt != 0)

comb, cmap = create_category_mask(init, gt)
if plot == 1: show_overview(init, gt, output, epe, rgb, comb, cmap, epethresh, title='', saveDir=False, show_now=True)
if plot == 2: show_plots(init, gt, output, epe, rgb, comb, cmap, epethresh)