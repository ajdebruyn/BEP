# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:40:16 2024

@author: sande
"""

import numpy as np

def initialize_halfstate(N, d, chi):
    """ Initializes the MPS into a product state of uniform eigenstates """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    
    Lambda_mat[:,0] = 1
    Gamma_mat[:,:,0,0] = 1/np.sqrt(d)
    
    #.locsize[:,:] = 1
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    return Gamma_mat, Lambda_mat, locsize

def initialize_flipstate(N, d, chi):
    """ Initializes the MPS into a product of alternating up/down states """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    
    Lambda_mat[:,0] = 1
    
    for i in range(0,N,2):
        Gamma_mat[i,0,0,0] = 1
    for i in range(1,N,2):
        Gamma_mat[i,d-1,0,0] = 1
           
    #.locsize[:,:] = 1
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    return Gamma_mat, Lambda_mat, locsize

def initialize_up_or_down(N, d, chi, up):
    """ Initializes the MPS into a product state of up or down states """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    if up:  #initialize each spin in up state
        i=0 
    else:   #initialize each spin in down state
        i=d-1
    Lambda_mat[:,0] = 1
    Gamma_mat[:,i,0,0] = 1
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    return Gamma_mat, Lambda_mat, locsize

def initialize_LU_RD(N, d, chi, scale_factor):
    """ Initializes the MPS linearly from up at the leftmost site to down at the rightmost site """
    """ scale_factor is a variable that defines the peak up/down values taken """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    
    Lambda_mat[:,0] = 1
    
    temp = 1-np.arange(N)/(N-1)
    temp = temp-0.5
    
    
    Gamma_mat[:,0,0,0] = np.sqrt(temp*scale_factor + 0.5)
    Gamma_mat[:,d-1,0,0] = np.sqrt(-1*temp*scale_factor + 0.5)
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    return Gamma_mat, Lambda_mat, locsize
 
