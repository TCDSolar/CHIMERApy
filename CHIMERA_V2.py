from astropy import wcs
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
import astropy.units as u
from astropy.visualization import astropy_mpl_style
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate
import sunpy
import sunpy.map
import sys
import sunpy.data.sample


plt.style.use(astropy_mpl_style)

# loading in the images as fits files

im171 = glob.glob('171.fts')
im193 = glob.glob('193.fts')
im211 = glob.glob('211.fts')
imhmi = glob.glob('hmi.fts')

#ensure that all images are present

if im171 == [] or im193 == [] or im211 == [] or imhmi == []:
	print("Not all required files present")
	sys.exit()

#Two functions that rescale the aia and hmi images from any original size to any final size

#didn't normalize by exposure time for hmi because it was equal to 0

def rescale_aia(image: np.array, orig_res: int, desired_res: int):
        hdu_number = 0
        hed = fits.getheader(image[0],hdu_number)
        dat= fits.getdata(image[0], ext=0)/(hed["EXPTIME"])
        if desired_res > orig_res:
            scaled_array=np.linspace(start = 0, stop = desired_res, num = orig_res)
            dn=scipy.interpolate.RectBivariateSpline(scaled_array,scaled_array,dat)
            if len(dn(np.arange(0, desired_res),np.arange(0,desired_res))) != desired_res:
                print("Incorrect image resolution")
                sys.exit()
            else:
                return dn(np.arange(0,desired_res),np.arange(0,desired_res))
        elif desired_res < orig_res:
            scaled_array=np.linspace(start = 0, stop = orig_res, num = desired_res)
            dn=scipy.interpolate.RectBivariateSpline(scaled_array,scaled_array,dat)
            if len(dn(np.arange(0, desired_res),np.arange(0,desired_res))) != desired_res:
                print("Incorrect image resolution")
                sys.exit()
            else:
                return dn(np.arange(0,desired_res),np.arange(0,desired_res))


def rescale_hmi(image: np.array, orig_res: int, desired_res: int):
        hdu_number = 0
        hed = fits.getheader(image[0],hdu_number)
        dat= fits.getdata(image[0], ext=0)
        if desired_res > orig_res:
            scaled_array=np.linspace(start = 0, stop = desired_res, num = orig_res)
            dn=scipy.interpolate.RectBivariateSpline(scaled_array,scaled_array,dat)
            if len(dn(np.arange(0, desired_res),np.arange(0,desired_res))) != desired_res:
                print("Incorrect image resolution")
                sys.exit()
            else:
                return dn(np.arange(0,desired_res),np.arange(0,desired_res))
        elif desired_res < orig_res:
            scaled_array=np.linspace(start = 0, stop = orig_res, num = desired_res)
            dn=scipy.interpolate.RectBivariateSpline(scaled_array,scaled_array,dat)
            if len(dn(np.arange(0, desired_res),np.arange(0,desired_res))) != desired_res:
                print("Incorrect image resolution")
                sys.exit()
            else:
                return dn(np.arange(0,desired_res),np.arange(0,desired_res))

#defining data and headers which are used in later steps
hdu_number = 0

data = rescale_aia(im171, 1024, 4096)
datb = rescale_aia(im193, 1024, 4096)
datc = rescale_aia(im211, 1024, 4096)
datm = rescale_hmi(imhmi, 1024, 4096)

heda=fits.getheader(im171[0],0)
hedb=fits.getheader(im193[0],0)
hedc=fits.getheader(im211[0],0)
hedm=fits.getheader(imhmi[0],0)

#filter_all: rescales 'cdelt1' 'cdelt2' 'cpix1' 'cipix2' if 'cdelt1' > 1
#filter_b: ensures 'ctype1' 'ctype2' are correctly defined as 'solar_x' and 'solar_y' respectively
#filter_hmi: rotates array if 'crota1' is greater than 90 degrees

def filter_all(aiaa: np.array, aiab: np.array, aiac: np.array):
	hdu_number = 0
	heda = fits.getheader(aiaa[0],hdu_number)
	hedb = fits.getheader(aiab[0],hdu_number)
	hedc = fits.getheader(aiac[0],hdu_number)
	if heda['cdelt1'] > 1:
		heda['cdelt1'],heda['cdelt2'],heda['crpix1'],heda['crpix2']=heda['cdelt1']/4.,heda['cdelt2']/4.,heda['crpix1']*4.0,heda['crpix2']*4.0
		hedb['cdelt1'],hedb['cdelt2'],hedb['crpix1'],hedb['crpix2']=hedb['cdelt1']/4.,hedb['cdelt2']/4.,hedb['crpix1']*4.0,hedb['crpix2']*4.0
		hedc['cdelt1'],hedc['cdelt2'],hedc['crpix1'],hedc['crpix2']=hedc['cdelt1']/4.,hedc['cdelt2']/4.,hedc['crpix1']*4.0,hedc['crpix2']*4.0

def filter_b(aiab: np.array):
	hdu_number = 0
	hedb = fits.getheader(aiab[0],hdu_number)
	if hedb["ctype1"] != 'solar_x ':
		hedb["ctype1"]='solar_x '
		hedb["ctype2"]='solar_y '

def filter_hmi(aiac: np.array):
	hdu_number = 0
	hedm=fits.getheader(imhmi[0],hdu_number)
	if hedm['crota1'] > 90:
		datm=np.rot90(np.rot90(datm))

filter_all(im171, im193, im211)
filter_hmi(imhmi)
filter_b(im193)


#removes negative values from an array
def remove_neg(aiaa: np.array, aiab:np.array, aiac: np.array):
	data[np.where(data <= 0)] = 0
	datb[np.where(datb <= 0)] = 0
	datc[np.where(datc <= 0)] = 0


remove_neg(data, datb, datc)

#defines shape of the array and the solar radius
def define_shape(aia: np.array):
    hdu_number = 0
    return np.shape(aia)

def define_radius(image: np.array):
    hdu_number = 0
    hed = fits.getheader(image[0],hdu_number)
    return hed['rsun']

#defining important variables
s = define_shape(data)
rs = define_radius(im171)
print(s)
print(rs)

#converting pixel values to arcsec
dattoarc = fits.getheader(im171[0],hdu_number)['cdelt1']
conver=(s[0]/2)*dattoarc/hedm['cdelt1']-(s[1]/2)
convermul = dattoarc/hedm['cdelt1']

#converts to the Heliographic Stonyhurst coordinate system

def to_helio(image: np.array):
    aia = sunpy.map.Map(image)
    adj = 4096/aia.dimensions[0].value
    x, y = (np.meshgrid(*[np.arange(adj*v.value) for v in aia.dimensions]) * u.pixel)/adj
    hpc = aia.pixel_to_world(x, y)
    return hpc.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)

hpc = to_helio(im171)
csys=wcs.WCS(hedb)

#setting up arrays to be used in later processing
#only difference between iarr and bmcool is integer vs. float?
ident = 1
iarr = np.zeros((s[0],s[1]),dtype=np.byte)
bmcool=np.zeros((s[0],s[1]),dtype=np.float32)

offarr,slate=np.array(iarr),np.array(iarr)
cand,bmmix,bmhot=np.array(bmcool),np.array(bmcool),np.array(bmcool)

#define the locations of the magnetic cutoffs
def cutoff_loc(size: int):
    r = (size[1]/2.0)-450
    xgrid,ygrid=np.meshgrid(np.arange(size[0]),np.arange(size[1]))
    center=[int(size[1]/2.0),int(size[1]/2.0)]
    return np.where((xgrid-center[0])**2+(ygrid-center[1])**2 > r**2)

#create 2D gaussian array for mag cutoffs
def create_gauss(size: int):
    y,x=np.mgrid[0:4096,0:4096]
    return Gaussian2D(1,size[0]/2,size[1]/2,2000/2.3548,2000/2.3548)(x,y)

w = cutoff_loc(s)
garr = create_gauss(s)
garr[w] = 1.0

#creates sub-arrays of props to isolate column of index 0 and column of index 1
#what is props??
props=np.zeros((26,30),dtype='<U16')
props[:,0]='ID','XCEN','YCEN','CENTROID','X_EB','Y_EB','X_WB','Y_WB','X_NB','Y_NB','X_SB','Y_SB','WIDTH','WIDTH°','AREA','AREA%','<B>','<B+>','<B->','BMAX','BMIN','TOT_B+','TOT_B-','<PHI>','<PHI+>','<PHI->'
props[:,1]='num','"','"','H°','"','"','"','"','"','"','"','"','H°','°','Mm^2','%','G','G','G','G','G','G','G','Mx','Mx','Mx'

#define threshold values in log s
def set_thresh(dat: np.array, b_val: float, u_val: float):
    with np.errstate(divide = 'ignore'):
        t = np.log10(dat)
    t[np.where(t < b_val)] = b_val
    t[np.where(t > u_val)] = u_val
    return np.array(((t - b_val)/(u_val - b_val))*255,dtype=np.float32)

t0 = set_thresh(datc, .8, 2.7)
t1 = set_thresh(datb, 1.4, 3.0)
t2 = set_thresh(data, 1.2, 3.9)

#ignores division and invalid erros in the following conditions to create 3 segmented bitmasks
with np.errstate(divide = 'ignore',invalid='ignore'):
	bmmix[np.where(t2/t0 >= ((np.mean(data)*0.6357)/(np.mean(datc))))]=1
	bmhot[np.where(t0+t1 < (0.7*(np.mean(datb)+np.mean(datc))))]=1
	bmcool[np.where(t2/t1 >= ((np.mean(data)*1.5102)/(np.mean(datb))))]=1

print(bmcool)
