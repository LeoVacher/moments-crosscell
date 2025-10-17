import sys
sys.path.append("./lib")

import numpy as np
import healpy as hp
import pymaster as nmt 
import urllib.request

fsky=0.99
scale=10
nside=64

fskylist=np.array([0.2,0.4,0.6,0.7,0.8,0.9,0.97,0.99])
field= list(np.where(fskylist==fsky)[0])

try :
    maskgal =hp.read_map("./masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits", field=field)
except:
    url = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
    output_file = "./masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
    urllib.request.urlretrieve(url, output_file)
    maskgal =hp.read_map("./masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits", field=field)

maskgal = hp.ud_grade(maskgal*1.,nside_out=nside)
maskgal[maskgal<0.5] = 0
maskgal[maskgal>=0.5] = 1
maskgalapo = nmt.mask_apodization(maskgal*1., scale, apotype='C2')

hp.mollview(maskgalapo)

hp.write_map("./masks/mask_fsky%s_nside%s_aposcale%s.npy"%(fsky,nside,scale), maskgalapo, dtype=float, overwrite=True)