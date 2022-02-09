# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:32:10 2022

@author: QQ
"""

from equations import SHEquation
import spectral 
import numpy as np

N_list = [16, 32, 64]
dtype_list = [np.complex128]


for dtype in dtype_list:    
    for N in N_list:
        x_basis = spectral.Fourier(N, interval=(0, 4*np.pi))
        domain = spectral.Domain([x_basis])
        x = x_basis.grid()
        u = spectral.Field(domain, dtype=dtype)
        u.require_grid_space()
        u.data = -2*np.cosh((x-2*np.pi))**(-2)

        KdV = SHEquation(domain, u)

        KdV.evolve(spectral.SBDF2, 1e-3, 1000)

        u.require_coeff_space()
        u.require_grid_space(scales=128//N)
        
        print("N = " + str(N))
        print("dtype = " + str(dtype))
        print(u.data[20])