import pandas as pd

import numpy as np
from PIL import Image
im = Image.open('data/nvdi/raw/MOD13A3.061__1_km_monthly_NDVI_doy2000032_aid0004.tif')
im.show()

x = np.array(im)
x = (x - x.min()) / (x.max() - x.min()) * 255
im2 = Image.fromarray(x)

import os
import matplotlib.pyplot as plt
import arcpy
arcpy.CheckOutExtension("ImageAnalyst")

class Nvdi:

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
