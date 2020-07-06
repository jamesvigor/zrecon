#!/usr/bin/env python
# coding: utf-8

# # Reconstruct .txrm Raw Data (Zeiss)

# The Zeiss xradia 410 versa produces output data of type txrm. The tomopy module can load these and then subsequently reconstruct as required, replacing the requirement for the Zeiss reconstructor on the workstation in the lab. 
# 
# To do this, we need the `io` submodule from the `dxchange` package, we will take `pyplot` from `matplotlib` for some visualisation, `numpy` for numerical analysis bits, and `tomopy` for the reconstruction bits.

# In[ ]:


import dxchange as io
import matplotlib.pyplot as plt
import numpy as np
import tomopy
from tomopy.prep import normalize
from tomopy.recon.rotation import write_center
from tomopy.recon.rotation import find_center
from tomopy.recon.algorithm import recon


# A couple of parameters are required here. The `slice_range` tuple contains the range of slices to load to memory in `(rotation, height, width)`. We then need to take some bits from the header to the image, which also contains the reference data and the dark current data.

# In[ ]:


# Load the data
data   = io.read_txrm("./C2_No acid.txrm", slice_range=(1601, 1, 2000))


# In[ ]:


data = np.copy(tdata) # RESET DATA, remove after debugging


# Take the header information from the dataset. Also, extract the reference data, set an arbitrary dark current image (see comment), set data to the raw data element in the dataset, and extract theta values for reconstruction.

# In[ ]:


header = data[1] # Header Values
ref    = data[1]['reference'] # Reference image values (already DC corrected)
arb_dark   = np.zeros_like(data[0][0]) # Dark current required for tomopy but corrections already applied on instrument.
data   = data[0] # Data values (already DC corrected)
theta  = header['thetas'] # Rotation position values


# Show the refrence image and counts in the reference.

# In[ ]:


print("Maximum counts in reference: {}".format(np.max(ref)))
print("Minimum counts in reference: {}".format(np.min(ref)))
plt.imshow(ref)


# Remove outliers from the dataset. Chunk along axis two (columns; in $x$ direction), perform n-1 median filtering along the other dimensions, and take a window size of 15 pixels. The outlier value is set to be a pixel 2000 raw counts above the median value of the subset. This can also run on a CUDA GPU if available. The reference image is a two axis image, raw data image is three axis.

# In[ ]:


#zinger_level = 2000
#data = tomopy.remove_outlier(data, zinger_level, size=15, axis=2)
#ref  = tomopy.remove_outlier(ref,  zinger_level, size=15, axis=1)

# CUDA code, should remain commented.
#data = tomopy.misc.corr.remove_outlier_cuda(data, zinger_level, size=15, axis=2)
#ref  = tomopy.misc.corr.remove_outlier_cuda(ref,  zinger_level, size=15, axis=2)


# Plot the output at rotation angle of 180 deg. 

# In[ ]:


plt.imshow(data[800])


# Normalize the dataset using the reference image `ref` and the arbitrary dark current image `arb_dark`. Run this on 4 cores and set the maximum transmission to 1.0. Then multiply the result by 100 to give a transmission percent value identical to those provided by the instrumentation then display output.

# In[ ]:


#cdata = np.copy(data)
data = tomopy.prep.normalize.normalize(data, ref, arb_dark, cutoff=1.0, ncore=4) * 100
plt.imshow(data[800])
print("Maximum Transmission: {}".format(np.round(np.max(data), 3)))
print("Minimum Transmission: {}".format(np.round(np.min(data), 3)))


# Remove stripes in the tomogram using the fourier wavelet method with 6 discrete wavelet transforms, a sym16 wavelet filter, a damping parameter of 2, and padding of the sinogram...

# In[ ]:


data = tomopy.prep.stripe.remove_stripe_fw(data, level=6, wname='sym16', sigma=2, pad=True, ncore=4, nchunk=10)


# Calculate phase contrast data, optionally add the propagation distance to the function using the `dist` variable. This can be determined from the original instrument parameters distance sample -> detector (not source -> detector - wave front propagation only occurs after refraction in the sample).

# In[ ]:


rat = 0.9
eng = 140
dist = 0
data = tomopy.prep.phase.retrieve_phase(data, pixel_size=header['pixel_size'], energy=eng, alpha=rat, pad=True)

# Propagation distance data...
#data = tomopy.prep.phase.retrieve_phase(data, pixel_size=header['pixel_size'], energy=eng, alpha=rat, pad=True, dist=dist)


# Find the center of rotation using artifacts present in a reconstructed image at discrete centres of rotation.

# In[ ]:


#CoR = find_center(data, theta, mask=True, tol=1)
CoR = write_center(data, theta, )
print("Center of Rotation found at: {}".format(CoR))


# Reconstruct the dataset using the algorithm specified. The Zeiss reconstructor probably uses the standard filtered back projection fbp algorithm, also available in tomopu are the blog algebraic reconstruction technique, and fourier grid reconstruction technique (probably quicker on a good workstation).

# In[ ]:


# Reconstruct using the fourier grid reconstruction algorithm...
rcn = recon(data[:,50:55,0:2000], theta, center=CoR, algorithm='gridrec', nchunk=100, ncore=4)

# Block algebraic reconstruction technique...
#rcn = recon(data[:,50:55,0:2000], theta, center=CoR, algorithm='bart', nchunk=100, ncore=4)

# Filtered back projection (probably what the Zeiss package does)...
#rcn = recon(data[:,50:51,0:2000], theta, center=CoR, algorithm='fbp', nchunk=100, ncore=4)


# The final thing to do is to write the reconstructed data to disk. This can then be visualised by some package like fiji / imagej or something more expensive.
