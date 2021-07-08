## RUN THIS PORTION for all dependent functions

import numpy as np
import matplotlib.pyplot as plt  
import os
from math import ceil, floor
from scipy.ndimage import gaussian_filter
from skimage.transform import resize, rescale

# function that is necessary to decode the UTF-8 storage of the dimensions for each FOV
def str_arr_to_float(str_array):
    str_convert = ""
    for i in range(len(str_array)):
        str_convert += str_array[i].decode('UTF-8')
    float_conv = float(str_convert)
    return float_conv

# was having issues with numpy broadcasting, so I did it manually and it seems to improve runtime
def col_mult(a,b):
    a = np.array(a)
    b = np.array(b)
    tmp_array = np.zeros(a.shape)
    for i in range(len(b)):
        tmp_array[:,i] = a[:,i]*b[i]
    return tmp_array

# was having issues with numpy broadcasting, so I did it manually and it seems to improve runtime
def row_mult(a,b):
    a = np.array(a)
    b = np.array(b)
    tmp_array = np.zeros(a.shape)
    for i in range(len(b)):
        tmp_array[i,:] = a[i,:]*b[i]
    return tmp_array

# was having issues with numpy broadcasting, so I did it manually and it seems to improve runtime
def col_sum(a,b):
    a = np.array(a)
    b = np.array(b)
    b = np.transpose(b)
    tmp_array = np.zeros(a.shape)
    for i in range(len(b)):
        tmp_array[:,i] = a[:,i]+b[i]
    return tmp_array

# was having issues with numpy broadcasting, so I did it manually and it seems to improve runtime
def row_sum(a,b):
    a = np.array(a)
    b = np.array(b)
    tmp_array = np.zeros(a.shape)
    for i in range(len(b)):
        tmp_array[i,:] = a[i,:]+b[i]
    return tmp_array

# translated version of matlab's quantile function
def quantile(x,q,dim = -1):
    if dim == 0:
        row, col = x.shape
        quant = np.zeros([row,1])
        for i in range(row):
            y = np.sort(x[i,:])
            quant[i,0] = np.interp(q, np.linspace(1/(2*col), (2*col-1)/(2*col), col), y)
    elif dim == 1:
        row, col = x.shape
        quant = np.zeros([1,col])
        for i in range(col):
            y = np.sort(x[:,i])
            quant[0,i] = np.interp(q, np.linspace(1/(2*row), (2*row-1)/(2*row), row), y)
    elif dim == -1:
        x = x.reshape([len(x),])
        n = len(x)
        y = np.sort(x)
        quant = (np.interp(q, np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))
    return quant

# translated component of matlab's quantile function
def spaced_quantiles(x,bins):
#     vals = range(1,bins+1)
#     percentiles = np.divide(vals,bins+1)
    vals = range(0,bins)
    percentiles = np.divide(vals,bins-1)
    bin_vals = np.array([quantile(x,q) for q in percentiles])
    return bin_vals

# necessary function to assess how linear algebra function is computed based on matlab implementation
def isSquare(m): 
    return all (len (row) == len (m) for row in m)

# implementation of histc from matlab, which counts number of terms in each bin
def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return r

# bread and butter step that computes the respective adjustments to normalize the histograms of the plots
def linhistmatch(a,b,nbins):
    a_nan_idx = np.isnan(a)
    b_nan_idx = np.isnan(b)
    a = a[a_nan_idx==False]
    b = b[b_nan_idx==False]
    abins = np.transpose(spaced_quantiles(a,nbins))
    bbins = np.transpose(spaced_quantiles(b,nbins))
    ones = np.ones(np.array([abins.shape[0],]))
    adj_abins = np.transpose(np.stack([abins, ones]))
    q,r = np.linalg.qr(adj_abins)
    beta = np.linalg.solve(r,np.matmul(np.linalg.pinv(q),bbins))
    atransform = np.full(a_nan_idx.shape, np.nan)
    atransform[a_nan_idx==False] = a*beta[0] + beta[1]
    return atransform, beta

# derives the values that will be applied to normalize the respective histograms
def vignette_correction(dataVolume, numbins = 400, numiter = 1, templatetype = 'middle_slice',visual = 0):
    dataVolume0=dataVolume
    Sh = []
    Dh = []
    Sv = []
    Dv = []
    for k in range(numiter):
        Xh = np.reshape(dataVolume,(dataVolume0.shape[0],-1),order='F')
        Xht = np.zeros(Xh.shape)
        Bh = np.zeros([Xh.shape[0],2])
        if templatetype.lower() == 'middle_slice':
            template = Xh[int(np.round(dataVolume0.shape[0]/2)),:]
            template = np.reshape(template, [1, len(template)],order='F')
        if templatetype.lower() == 'middle_20_slice':
            middle = int(np.round(dataVolume0.shape[0]/2))
            middle_20 = range(middle-10,middle+11)
            template = Xh[middle_20,:]
            template = np.reshape(template,[1, -1],order='F')
        if templatetype.lower() == 'random':
            numel = Xh.size
            r = np.random.permutation(numel)
            Xh_flattened = np.reshape(Xh,[-1,1],order='F')
            template = Xh_flattened[r[0:Xh.shape[1]]]
            template = np.reshape(template,[1, -1],order='F')

        for i in range(Xh.shape[0]):
            Xht[i,:], Bh[i,:] = linhistmatch(Xh[i,:],template,numbins)
        Sh.append(Bh[:,0])
        Dh.append(Bh[:,1])
        new_dataVolume = np.reshape(Xht, dataVolume0.shape,order='F')
        new_dataVolume = np.flip(new_dataVolume, 2)
        

        Xv = np.transpose(np.reshape(np.transpose(new_dataVolume,(1,0,2)),(new_dataVolume.shape[1],-1),order='F'))
        Xvt = np.zeros(Xv.shape)
        Bv = np.zeros([2,Xv.shape[1]])
        if templatetype.lower() == 'middle_slice':
            template = Xv[:,int(np.round(new_dataVolume.shape[0]/2))]
            template = np.reshape(template, [1, len(template)],order='F')

        if templatetype.lower() == 'middle_20_slice':
            middle = int(np.round(new_dataVolume.shape[0]/2))
            middle_20 = range(middle-10,middle+11)
            template = Xv[:,middle_20]
            template = np.reshape(template,[1, -1],order='F')

        if templatetype.lower() == 'random':
            numel = Xv.size
            r = np.random.permutation(numel)
            Xv_flattened = np.reshape(Xv,[-1,1],order='F')
            template = Xv_flattened[r[0:Xv.shape[0]]]
            template = np.reshape(template,[1, -1],order='F')
        for i in range(Xv.shape[1]):
            Xvt[:,i], Bv[:,i] = linhistmatch(Xv[:,i],template,numbins)
        Sv.append(Bv[0,:])
        Dv.append(Bv[1,:])
        
        dataVolume = np.reshape(np.transpose(Xvt),dataVolume0.shape,order='F')
    s = dataVolume.shape[0]/5
    vfield_corrected = np.transpose(gaussian_filter(np.max(dataVolume, axis = 2), sigma=s,mode = 'nearest',truncate=2.0))
    vfield = np.transpose(gaussian_filter(np.max(dataVolume0,axis =2), sigma=s,mode = 'nearest',truncate=2.0))
        
    Sh = np.array(Sh)
    Sv = np.array(Sv)
    Dh = np.array(Dh)
    Dv = np.array(Dv)
    return dataVolume,Sh, Sv, Dh, Dv, vfield_corrected, vfield

# loading the data into a dataframe
def data_loader(myPath, downsample_factor = 1, downslice_factor = 1):
    num_files = len([f for f in os.listdir(myPath)
         if f.endswith('.npy') and os.path.isfile(os.path.join(myPath, f))])
    files = []
    z_depths = []
    positions = {}
    t = 0
    for file in sorted(os.listdir(myPath)):
        if file.endswith(".npy"):
            filename = os.path.join(myPath,file)
            files.append(filename)
            im = np.load(filename)
            if t == 0:
                num_chans = im.shape[3]
                x_int = im.shape[0]
                y_int = im.shape[1]
                z_int = im.shape[2]
                t = 1
                r = np.random.permutation(z_int) - 1
                slicesamples=r[0:ceil(len(r)/downslice_factor)]
                if slicesamples.size == 0:
                    slicesamples = [0]
                dataVolume=np.zeros([int(np.round(x_int/downsample_factor)),int(np.round(y_int/downsample_factor)),num_files*len(slicesamples),num_chans])
            for k, s in enumerate(slicesamples):
                for c in range(num_chans):
                    ds = im[:,:,k,c]
                    ds = np.array(ds)
                    lo_threshold = np.percentile(ds,0.05) / 2
                    up_threshold = np.percentile(ds,99.95) * 2
                    ds = np.clip(ds,lo_threshold,up_threshold)
                    ds = np.transpose(ds)
                    dataVolume[:,:,k,c] = resize(ds,(int(np.round(x_int/downsample_factor)),int(np.round(y_int/downsample_factor))), preserve_range=True)
    return x_int, y_int, z_int, num_chans, files, positions, dataVolume

# applies the derived vignette correction onto the individual images
def apply_vignette_correction(file,x_int,y_int,z_int,nSh,nDh,nSv,nDv):
    im = np.load(file)
    num_chans = im.shape[3]
    raw_stack = np.zeros((num_chans,z_int,x_int,y_int))
    corr_stack = np.zeros((num_chans,z_int,x_int,y_int))
    numiter = nSh[0].shape[0]
    for z in range(z_int):            
        for i in range(num_chans):
            raw_ds = np.transpose(np.array(im[:,:,z,i]))
            lo_threshold = np.percentile(raw_ds,0.05) / 2
            up_threshold = np.percentile(raw_ds,99.95) * 2
            raw_stack[i,z,:,:] = np.clip(raw_ds,lo_threshold,up_threshold)
            corr_ds = np.clip(raw_ds,lo_threshold,up_threshold)
            for k in range(numiter):
                corr_ds = col_sum(col_mult(corr_ds,nSh[i][k]),nDh[i][k])
                corr_ds = row_sum(row_mult(corr_ds,nSv[i][k]),nDv[i][k])
            corr_stack[i,z,:,:] = corr_ds  
    return corr_stack, raw_stack