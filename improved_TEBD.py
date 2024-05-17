# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:27:49 2024

@author: sande
"""



import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from scipy.linalg import expm

import matplotlib.pyplot as plt

import pickle
import time
from datetime import datetime
import math

from initializations_mps import initialize_halfstate, initialize_LU_RD


class MPS:
    def __init__(self, ID, N, d, chi, is_density):
        self.ID = ID
        self.N = N
        self.d = d
        self.chi = chi
        self.is_density = is_density
        if is_density:
            self.name = "DENS"+str(ID)
        else: 
            self.name = "MPS"+str(ID)
        
        self.Lambda_mat = np.zeros((N+1,chi))
        self.Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)

        self.locsize = np.zeros(N+1, dtype=int)     #locsize tells us which slice of the matrices at each site holds relevant information
        
        self.spin_current_values = np.array([])
        self.normalization = np.array([])
        self.trace = np.array([])
        return
        
    def __str__(self):
        if self.is_density:
            return f"Density matrix {self.ID}, {self.N} sites of dimension {self.d}, chi={self.chi}"
        else:
            return f"MPS {self.ID}, {self.N} sites of dimension {self.d}, chi={self.chi}"
            
    def store(self):
        """ Stores the object to memory using pickle """
        time = str(datetime.now())
        timestr = time[5:7] + time[8:10] + "_" + time[11:13] + time[14:16] + "_"  #get month, day, hour, minute
        
        folder = "data\\" 
        filename = timestr+self.name+"_N"+str(self.N)+"_chi"+str(self.chi)+".pkl"
        
        file = open(folder + filename, 'wb')
        pickle.dump(self, file)
        
        print(f"Stored {filename} to memory")
        pass        
      
    
    def construct_vidal_supermatrices(self, newchi):
        """ Constructs a superket of the density operator in Vidal decomposition """
        sup_Gamma_mat = np.zeros((self.N, self.d**2, newchi, newchi), dtype=complex)
        sup_Lambda_mat = np.zeros((self.N+1, newchi))
        for i in range(self.N):
            sup_Gamma_mat[i,:,:,:] = np.kron(self.Gamma_mat[i], np.conj(self.Gamma_mat[i]))[:,:newchi,:newchi]
            sup_Lambda_mat[i,:] = np.kron(self.Lambda_mat[i], self.Lambda_mat[i])[:newchi]
        sup_Lambda_mat[N,:] = np.kron(self.Lambda_mat[N], self.Lambda_mat[N])[:newchi]
        sup_locsize = np.minimum(self.locsize**2, newchi)
        return sup_Gamma_mat, sup_Lambda_mat, sup_locsize
    
    def contract(self, begin, end):
        """ Contracts the gammas and lambdas between sites 'begin' and 'end' """
        theta = np.diag(self.Lambda_mat[begin,:]).copy()
        theta = theta.astype(complex)
        for i in range(end-begin+1):
            theta = np.tensordot(theta, self.Gamma_mat[begin+i,:,:,:], axes=(-1,1)) #(chi,...,d,chi)
            theta = np.tensordot(theta, np.diag(self.Lambda_mat[begin+i+1]), axes=(-1,1)) #(chi,...,d,chi)
        theta = np.rollaxis(theta, -1, 1) #(chi, chi, d, ..., d)
        return theta
    
    def decompose_contraction(self, theta, i, normalize):
        """ decomposes a given theta back into Vidal decomposition. i denotes the leftmost site contracted into theta """
        num_sites = np.ndim(theta)-2 # The number of sites contained in theta
        temp = num_sites-1           # Total number of loops required
        for j in range(temp):
            theta = theta.reshape((self.chi, self.chi, self.d, self.d**(temp-j)))
            theta = theta.transpose(2,0,3,1) #(d, chi, d**(temp-j), chi)
            theta = theta.reshape((self.d*self.chi, self.d**(temp-j)*self.chi))
            X, Y, Z = np.linalg.svd(theta); Z = Z.T
            #This can be done more efficiently by leaving out the Z=Z.T and only doing so in case of j==2
            
            if normalize==True:
                self.Lambda_mat[i+j+1,:] = Y[:self.chi] *1/np.linalg.norm(Y[:self.chi])
            else:
                self.Lambda_mat[i+j+1,:] = Y[:self.chi]
            
            X = np.reshape(X[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))
            inv_lambdas = self.Lambda_mat[i+j, :self.locsize[i+j]].copy()
            inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
            X = np.tensordot(np.diag(inv_lambdas),X[:,:self.locsize[i+j],:self.locsize[i+j+1]],axes=(1,1)) #(chi, d, chi)
            X = X.transpose(1,0,2)
            self.Gamma_mat[i+j, :, :self.locsize[i+j],:self.locsize[i+j+1]] = X

            theta_prime = np.reshape(Z[:self.chi*self.d**(temp-j),:self.chi], (self.d**(temp-j), self.chi, self.chi))
            if j==(temp-1):
                theta_prime = theta_prime.transpose(0,2,1)
                inv_lambdas  = self.Lambda_mat[i+j+2, :self.locsize[i+j+2]].copy()
                inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
                tmp_gamma = np.tensordot(theta_prime[:,:self.locsize[i+j+1],:self.locsize[i+j+2]], np.diag(inv_lambdas), axes=(2,0)) #(d, chi, chi)
                self.Gamma_mat[i+j+1, :, :self.locsize[i+j+1],:self.locsize[i+j+2]] = tmp_gamma 
            else:
                theta_prime = theta_prime.transpose(1,2,0)
                #Here we must contract Lambda with V for the next SVD. The contraction runs over the correct index (the chi resulting from the previous SVD, not the one incorporated with d**(temp-j))
                theta_prime = np.tensordot(np.diag(Y[:chi]), theta_prime, axes=(1,1))
        return
    
    def apply_singlesite(self, TimeOp, i):
        """ Applies a single-site operator to site i """
        theta = self.contract(i,i)
        theta_prime = np.tensordot(theta, TimeOp, axes=(2,1)) #(chi, chi, d)
        
        inv_lambdas  = self.Lambda_mat[i].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        theta_prime = np.tensordot(np.diag(inv_lambdas), theta_prime, axes=(1,0)) #(chi, chi, d) 
        
        inv_lambdas = self.Lambda_mat[i+1].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        theta_prime = np.tensordot(theta_prime, np.diag(inv_lambdas), axes=(1,0)) #(chi, d, chi)
        self.Gamma_mat[i,:,:,:] = np.transpose(theta_prime, (1,0,2))
        return

    def apply_twosite(self, TimeOp, i, normalize):
        """ Applies a two-site operator to sites i and i+1 """
        theta = self.contract(i,i+1) #(chi, chi, d, d)
        #operator is applied, tensor is reshaped
        theta_prime = np.tensordot(theta,TimeOp,axes=([2,3],[2,3])) #(chi,chi,d,d)        
        #self.decompose_contraction(theta_prime, i, normalize)
        
        #Note: using the decompose_contraction function is overkill for decomposing a twosite contraction
        #      therefore the code is repeated here
        #"""
        theta_prime = np.reshape(np.transpose(theta_prime, (2,0,3,1)),(self.d*self.chi, self.d*self.chi)) #first to (d, chi, d, chi), then (d*chi, d*chi)
        X, Y, Z = np.linalg.svd(theta_prime); Z = Z.T
        if normalize:
            self.Lambda_mat[i+1,:] = Y[:self.chi]*1/np.linalg.norm(Y[:self.chi])
        else:
            self.Lambda_mat[i+1,:] = Y[:self.chi]
        
        #truncation, and multiplication with the inverse lambda matrix of site i, where care is taken to avoid divides by 0
        X = np.reshape(X[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi)) 
        inv_lambdas  = self.Lambda_mat[i, :self.locsize[i]].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(np.diag(inv_lambdas),X[:,:self.locsize[i],:self.locsize[i+1]],axes=(1,1)) #(chi, d, chi)
        self.Gamma_mat[i, :, :self.locsize[i],:self.locsize[i+1]] = np.transpose(tmp_gamma,(1,0,2))
        
        #truncation, and multiplication with the inverse lambda matrix of site i+2, where care is taken to avoid divides by 0
        Z = np.reshape(Z[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))
        Z = np.transpose(Z,(0,2,1))
        inv_lambdas = self.Lambda_mat[i+2, :self.locsize[i+2]].copy()
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(Z[:,:self.locsize[i+1],:self.locsize[i+2]], np.diag(inv_lambdas), axes=(2,0)) #(d, chi, chi)
        self.Gamma_mat[i+1, :, :self.locsize[i+1],:self.locsize[i+2]] = tmp_gamma
        #"""
        return 

        
    def TEBD(self, TimeOp, Diss_arr, normalize, Diss_bool):
        """ TEBD algorithm """

        for i in range(0, self.N-1, 2):
            self.apply_twosite(TimeOp[i]/2, i, normalize)
        for i in range(1, self.N-2, 2):
            self.apply_twosite(TimeOp[i], i, normalize)
        
        if Diss_bool:
            if self.N%2 == 1:
                self.apply_twosite(TimeOp[self.N-2]/2, self.N-2, normalize)
            for i in range(len(Diss_arr["index"])):
                self.apply_singlesite(Diss_arr["TimeOp"][i], Diss_arr["index"][i])
            if self.N%2 == 1:
                self.apply_twosite(TimeOp[self.N-2]/2, self.N-2, normalize)
        
        elif self.N%2 == 1:
            self.apply_twosite(TimeOp[self.N-2], self.N-2, normalize)
        
        for i in range(0, self.N-1, 2):
            self.apply_twosite(TimeOp[i]/2, i, normalize)
        
        return
        
    def expval(self, Op, site):
        """ Calculates the expectation value of an operator Op for a single site """
        if self.is_density:     #In case of density matrices we must take the trace  
            Gamma_temp = self.Gamma_mat[site].copy()
            self.apply_singlesite(Op, site)
            result = self.calculate_vidal_inner(NORM_state)
            self.Gamma_mat[site] = Gamma_temp
            return result
        else:
            theta = self.contract(site,site) #(chi, chi, d)
            theta_prime = np.tensordot(theta, Op, axes=(2,1)) #(chi, chi, d)
            return np.real(np.tensordot(theta_prime, np.conj(theta), axes=([0,1,2],[0,1,2])))
    
    def expval_chain(self, Op):
        """ calculates expectation value for operator Op for the entire chain """
        result = np.zeros(self.N)
        for i in range(self.N):
            result[i] = self.expval(Op, i)
        return result
        
    def expval_twosite(self, Op, site):
        """ Calculates expectation value for a twosite operator Op at sites site and site+1 """
        if self.is_density:
            temp_Lambda = self.Lambda_mat[site+1].copy()
            temp_Gamma = self.Gamma_mat[site:site+2].copy()
            Op = Op.reshape(self.d,self.d,self.d,self.d)
            self.apply_twosite(Op,site,False)
            result = self.calculate_vidal_inner(NORM_state)
            self.Lambda_mat[site+1] = temp_Lambda
            self.Gamma_mat[site:site+2] = temp_Gamma
            return result
        else:
            theta = self.contract(site, site+1)
            Op = np.reshape(Op, (self.d,self.d,self.d,self.d))
            theta_prime = np.tensordot(theta, Op,axes=([2,3],[2,3])) #(chi,chi,d,d) 
            return np.real(np.tensordot(theta_prime, np.conj(theta), axes=([0,1,2,3],[0,3,1,2])))
   
    def calculate_energy(self, TimeEvol_obj):
        """ calculate the energy of the entire chain from a given Hamiltonian """
        Energy = 0
        for i in range(self.N-1):
            Energy += self.expval_twosite(TimeEvol_obj.Ham_energy[i],i)
        return Energy
    
    def calculate_vidal_inner(self, MPS2):
        """ Calculates the inner product of the MPS with another MPS """
        m_total = np.eye(self.chi)
        temp_gammas, temp_lambdas = MPS2.Gamma_mat, MPS2.Lambda_mat  #retrieve gammas and lambdas of MPS2
        for j in range(0, self.N):
            st1 = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(2,0)) #(d, chi, chi)
            st2 = np.tensordot(temp_gammas[j,:,:,:],np.diag(temp_lambdas[j+1,:]), axes=(2,0)) #(d, chi, chi)
            m_total = np.tensordot(m_total, np.conj(st1), axes=(0,1)) #(chi, d, chi)
            m_total = np.tensordot(m_total, st2, axes=([1,0],[0,1])) #(chi, chi)
        return np.real(m_total[0,0])
    
    
    def time_evolution(self, TimeEvol_obj, normalize, steps, desired_expectations, track_normalization, track_energy, track_current):
        if TimeEvol_obj.is_density != self.is_density:
            print("Error: time evolution operator type does not match state type (MPS/DENS)")
            return
        if (track_normalization==False and self.is_density==True):
            print("Tracking normalization is required for calculating expectation values in DENS foralism")
            print("Be aware this may cause issues")
            
        
        #### Initializing operators and expectation value arrays
        TimeOp = TimeEvol_obj.TimeOp
        Diss_arr = TimeEvol_obj.Diss_arr
        Diss_bool = TimeEvol_obj.Diss_bool
        
        Sz_expvals = np.zeros((self.N, steps),dtype=float)
        if track_energy:
            energy = np.zeros(steps)
                            
        #### Time evolution steps
        print(f"Starting time evolution of {self.name}")
        for t in range(steps):
            if (t%20==0):
                print(t)
            
            
            if track_normalization:
                self.normalization = np.append(self.normalization, self.calculate_vidal_inner(self))
                if self.is_density:
                    self.trace = np.append(self.trace, self.calculate_vidal_inner(NORM_state))
            if track_energy:
                energy[t] = self.calculate_energy(TimeEvol_obj)
            if (track_current==True and Diss_bool==True):
               middle_site = int(np.round(self.N/2-1))
               self.spin_current_values = np.append(self.spin_current_values, np.real( self.expval_twosite(spin_current_op, middle_site) ))
               if self.is_density:
                    self.spin_current_values[-1] *= 1/self.trace[t]
            
            if self.is_density:
                for i in range(self.N):
                    Sz_expvals[i,t] = self.expval(np.kron(Sz, np.eye(d)), i)
                Sz_expvals[:,t] *= 1/self.trace[t]
            else:
                for i in range(self.N):
                    Sz_expvals[i,t] = self.expval(Sz, i)
                
            self.TEBD(TimeOp, Diss_arr, normalize, Diss_bool)
 
        #### Plotting expectation values
        
        time_axis = np.arange(steps)*abs(TimeEvol_obj.dt)
        
        for i in range(self.N):
            plt.plot(time_axis, Sz_expvals[i,:], label="Site "+str(i))
        plt.title(f"<Sz> of {self.name} over time")
        plt.xlabel("Time")
        plt.ylabel("Sz")
        plt.grid()
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.show()
        
        if track_normalization:
            plt.plot(time_axis, self.normalization[-steps:])
            plt.title(f"Normalization of {self.name} over time")
            plt.xlabel("Time")
            plt.ylabel("Normalization")
            plt.grid()
            plt.show()
            if self.is_density:
                plt.plot(time_axis, self.trace[-steps:])
                plt.title(f"Trace of {self.name} over time")
                plt.xlabel("Time")
                plt.ylabel("Trace")
                plt.grid()
                plt.show()
        if track_energy:
            plt.plot(time_axis, energy)
            plt.title(f"Energy of {self.name}")
            plt.xlabel("Time")
            plt.ylabel("Energy")
            plt.grid()
            plt.show()
        
        if (track_current==True and Diss_bool==True):
            plt.plot(self.spin_current_values)
            plt.title(f"Current of {self.name} over time")
            plt.xlabel("Last x timesteps")
            plt.ylabel("Current")
            plt.grid()
            plt.show()
            
        return
    

########################################################################################  

class Time_Operator:
    def __init__(self,N, d, JXY, JZ, h, s_coup, dt, Diss_bool, is_density, use_CN):
        self.N = N
        self.d = d
        self.JXY = JXY
        self.JZ = JZ
        self.h = h
        self.s_coup = s_coup
        self.dt = dt
        self.is_density = is_density
        self.Diss_bool = Diss_bool
        self.use_CN = use_CN
        
        if isinstance(dt, complex):     
            self.is_imaginary = True
        else:
            self.is_imaginary = False
        
        if self.is_density==False:
            self.Diss_bool=False
       
        #### Creating Hamiltonian and Time operators
        #### Note: Ham_energy is the Hamiltonian to be used for energy calculation
        if self.is_density:
            self.Ham, self.Ham_energy = self.Create_Dens_Ham()
        else:
            self.Ham = self.Create_Ham()
            self.Ham_energy = self.Ham
        
        self.TimeOp = self.Create_TimeOp(self.dt, self.use_CN)
        
        if (self.is_density and self.Diss_bool):
            self.Diss_arr = self.Create_Diss_Array(self.s_coup)
            self.Calculate_Diss_TimeOp(self.dt, self.use_CN)
        else:
            self.Diss_arr = None
        return
        
    def Create_Ham(self):
        """ Create Hamiltonian for purestate """
        SX = np.kron(Sx, Sx)
        SY = np.kron(Sy, Sy)
        SZ = np.kron(Sz, Sz)
        SZ_L = np.kron(Sz, np.eye(self.d))
        SZ_R = np.kron(np.eye(self.d), Sz)
        SZ_M = (SZ_L + SZ_R)
        
        H_L = self.h*(SZ_L + SZ_R/2) + self.JXY*(SX + SY) + self.JZ*SZ
        H_R = self.h*(SZ_L/2 + SZ_R) + self.JXY*(SX + SY) + self.JZ*SZ
        H_M = self.h*SZ_M/2 + self.JXY*(SX + SY) + self.JZ*SZ
        
        H_arr = np.ones((self.N-1, self.d**2, self.d**2), dtype=complex)
        
        H_arr[1:self.N-2,:,:] *= H_M
        H_arr[0,:,:] = H_L
        H_arr[self.N-2,:,:] = H_R
        return H_arr
        
    def Create_Dens_Ham(self):
        """ create effective Hamiltonian for time evolution of the density matrix """
        Sx_arr = np.array([np.kron(Sx, np.eye(self.d)) , np.kron(np.eye(self.d), Sx)])
        Sy_arr = np.array([np.kron(Sy, np.eye(self.d)) , np.kron(np.eye(self.d), Sy)])
        Sz_arr = np.array([np.kron(Sz, np.eye(self.d)) , np.kron(np.eye(self.d), Sz)])
         
        H_arr = np.ones((2, N-1, self.d**4, self.d**4), dtype=complex)
        for i in range(2):
            SX = np.kron(Sx_arr[i], Sx_arr[i])
            SY = np.kron(Sy_arr[i], Sy_arr[i])
            SZ = np.kron(Sz_arr[i], Sz_arr[i])
            SZ_L = np.kron(Sz_arr[i], np.eye(self.d**2))
            SZ_R = np.kron(np.eye(self.d**2), Sz_arr[i])
            SZ_M = (SZ_L + SZ_R)
            
            H_L = self.h*(SZ_L + SZ_R/2) + self.JXY*(SX + SY) + self.JZ*SZ
            H_R = self.h*(SZ_L/2 + SZ_R) + self.JXY*(SX + SY) + self.JZ*SZ
            H_M = self.h*SZ_M/2 + self.JXY*(SX + SY) + self.JZ*SZ
       
            H_arr[i, 1:self.N-2,:,:] *= H_M
            H_arr[i, 0,:,:] = H_L
            H_arr[i, self.N-2,:,:] = H_R

        #Note: H_arr[0] is the correct Hamiltonian to use for energy calculations
        return (H_arr[0] - np.conj(H_arr[1])), H_arr[0]     

    def Create_TimeOp(self, dt, use_CN):
        if self.is_density:
            U = np.ones((self.N-1, self.d**4, self.d**4), dtype=complex)
        else:
            U = np.ones((self.N-1, self.d**2, self.d**2), dtype=complex)
        
        if use_CN:
            U[0,:,:] = self.create_crank_nicolson(self.Ham[0], dt)
            U[self.N-2,:,:] = self.create_crank_nicolson(self.Ham[self.N-2], dt)
            U[1:self.N-2,:,:] *= self.create_crank_nicolson(self.Ham[1], dt) # we use broadcasting
        else:
            U[0,:,:] = expm(-1j*dt*self.Ham[0])
            U[self.N-2,:,:] = expm(-1j*dt*self.Ham[self.N-2])
            U[1:self.N-2,:,:] *= expm(-1j*dt*self.Ham[1]) # we use broadcasting
    
        U = np.around(U, decimals=15)        #Rounding out very low decimals 
        if self.is_density:
            return np.reshape(U, (self.N-1,self.d**2,self.d**2,self.d**2,self.d**2))
        else:
            return np.reshape(U, (self.N-1,self.d,self.d,self.d,self.d)) 

    def create_crank_nicolson(self, H, dt):
        """ Creates the Crank-Nicolson operator from a given Hamiltonian """
        H_top=np.eye(H.shape[0])-1j*dt*H/2
        H_bot=np.eye(H.shape[0])+1j*dt*H/2
        return np.linalg.inv(H_bot).dot(H_top)

    def Calculate_Diss_site(self, Lind_Op):
        """ Creates the dissipative term for a single site """
        """ Lind_Op is shape (k,d,d) or (d,d) -- the k-index is in case multiple different lindblad operators act on a single site """
        Diss = np.zeros((self.d**2, self.d**2), dtype=complex)
        if Lind_Op.ndim==2:     #If only a single operator is given, this matrix is used
            Diss += 2*np.kron(Lind_Op, np.conj(Lind_Op))
            Diss -= np.kron(np.matmul(np.conj(np.transpose(Lind_Op)), Lind_Op), np.eye(self.d))
            Diss -= np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op), np.conj(Lind_Op)))
        else:                   #If multiple matrices are given, the sum of Lindblad operators is used
            for i in range(np.shape(Lind_Op)[0]):
                Diss += 2*np.kron(Lind_Op[i], np.conj(Lind_Op[i]))
                Diss -= np.kron(np.matmul(np.conj(np.transpose(Lind_Op[i])), Lind_Op[i]), np.eye(self.d))
                Diss -= np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op[i]), np.conj(Lind_Op[i])))
        return Diss
    
    def Create_Diss_Array(self, s_coup):
        """ Creates the array containing dissipative term, where 'index' stores the site the corresponding Lindblad operators couple to """
        Diss_arr = np.zeros((), dtype=[
            ("index", int, 2),
            ("Operator", complex, (2, self.d**2, self.d**2)),
            ("TimeOp", complex, (2, self.d**2, self.d**2))
            ])
        
        Diss_arr["index"][0] = 0
        Diss_arr["Operator"][0,:,:] = self.Calculate_Diss_site(s_coup*Sp)
    
        #Diss_arr["index"][1] = N-1
        #Diss_arr["Operator"][1,:,:] = self.Calculate_Diss_site(np.sqrt(2*s_coup)*np.eye(self.d))
    
        Diss_arr["index"][1] = N-1
        Diss_arr["Operator"][1,:,:] = self.Calculate_Diss_site(s_coup*Sm)
        return Diss_arr
    
    def Calculate_Diss_TimeOp(self, dt, use_CN):
        """ Calculates the dissipative time evolution operators """
        for i in range(len(self.Diss_arr["index"])):
            if use_CN:
                temp = self.create_crank_nicolson(self.Diss_arr["Operator"][i], dt)
            else:
                temp = expm(dt*self.Diss_arr["Operator"][i])
            temp = np.around(temp, decimals=15)    #Rounding out very low decimals 
            self.Diss_arr["TimeOp"][i,:,:] = temp
        return



###################################################################################

def load_state(folder, name, new_ID):
    """ loads a pickled state from folder 'folder' with name 'name' - note: name must include .pkl """
    filename = folder + name
    with open(filename, 'rb') as file:  
        loaded_state = pickle.load(file)
    globals()[loaded_state.name] = loaded_state
    
    loaded_state.ID = new_ID
    if loaded_state.is_density:
        loaded_state.name = "DENS"+str(new_ID)
    else: 
        loaded_state.name = "MPS"+str(new_ID)
    return loaded_state
    

def create_superket(State, newchi):
    """ create MPS of the density matrix of a given MPS """
    gammas, lambdas, locsize = State.construct_vidal_supermatrices(newchi)
    
    name = "DENS" + str(State.ID)
    newDENS = MPS(State.ID, State.N, State.d**2, newchi, True)
    newDENS.Gamma_mat = gammas
    newDENS.Lambda_mat = lambdas
    newDENS.locsize = locsize
    globals()[name] = newDENS
    return newDENS

def create_maxmixed_normstate():
    """ Creates vectorised density matrix of an unnormalized maximally mixed state, used to calculate the trace of a vectorised density matrix """
    """ since to obtain rho11 + rho22 you must take inner [1 0 0 1] [rho11 rho12 rho21 rho22]^T without a factor 1/sqrt(2) in front """
    lambdas = np.zeros((N+1,newchi))
    lambdas[:,0]= 1
    
    gammas = np.zeros((N,d**2,newchi,newchi), dtype=complex)
    diagonal = (1+d)*np.arange(d)
    gammas[:,diagonal, 0, 0] = 1        #/2  #/np.sqrt(2)
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,newchi**2)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum((d**2)**arr, newchi**2)
    
    NORM_state = MPS(0, N, d**2, newchi, True)
    NORM_state.Gamma_mat = gammas
    NORM_state.Lambda_mat = lambdas
    NORM_state.locsize = locsize
    return NORM_state

    
def calculate_thetas_singlesite(state):
    """ contracts lambda_i gamma_i lambda_i+1 (:= theta) for each site and returns them, used for the NORM_state """
    """ NOTE: only works for NORM_state since there the result is the same for all sites! """
    """ This function is used to prevent redundant calculation of these matrices """
    #Note, the lambda matrices are just a factor 1, it is possible to simply return a reshaped gamma matrix
    #temp = np.tensordot(np.diag(state.Lambda_mat[0,:]), state.Gamma_mat[0,:,:,:], axes=(1,1)) #(chi, d, chi)
    #return np.tensordot(temp, np.diag(state.Lambda_mat[1,:]),axes=(2,0)) #(chi,d,chi)
    return state.Gamma_mat[0].transpose(0,2,1)

def calculate_thetas_twosite(state):
    """ contracts lambda_i gamma_i lambda_i+1 gamma_i+1 lambda_i+2 (:= theta) for each site and returns them, used for the NORM_state """
    """ NOTE: only works for NORM_state since there the result is the same for all sites! """
    """ This function is used to prevent redundant calculation of these matrices """
    temp = np.tensordot(np.diag(state.Lambda_mat[0,:]), state.Gamma_mat[0,:,:,:], axes=(1,1)) #(chi, d, chi)
    temp = np.tensordot(temp,np.diag(state.Lambda_mat[1,:]),axes=(2,0)) #(chi, d, chi) 
    temp = np.tensordot(temp, state.Gamma_mat[1,:,:,:],axes=(2,1)) #(chi, d, chi, d) -> (chi,d,d,chi)
    return np.tensordot(temp,np.diag(state.Lambda_mat[2,:]), axes=(3,0)) #(chi, d, d, chi)







####################################################################################
t0 = time.time()
#### Simulation variables
N=8         #System Length
d=2         #Spin dimension -- do not change
chi=20      #MPS truncation parameter
newchi=40   #DENS truncation parameter

im_steps = 0            #Number of timesteps in imaginary time
im_dt = -0.03j          #Size of imaginary timestep
steps=1000               # '' for real timestep
dt = 0.02               # '' for real timestep

normalize = True        #whether to maintain MPS normalization
use_CN = False          #to use Crank-Nicolson approximation
Diss_bool = True       #to implement dissipative terms


#### Hamiltonian and Lindblad constants
h=0                         #Strength of the transverse magnetic field
JXY=1                       #XY Coupling strength XXZ Hamiltonian
JZ=1                      #Z coupling strength of XXZ Hamiltonian

s_coup=1                    #Coupling strength of Dissipative terms
s_coup = np.sqrt(s_coup)  


#### Spin matrices
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])


#### Spin current operator and cutoff factor
spin_current_op = 2 * ( np.kron( np.kron(Sx, np.eye(d)) , np.kron(Sy, np.eye(d))) - np.kron( np.kron(Sy, np.eye(d)) , np.kron(Sx, np.eye(d))) )
#equivalent operator in terms of Sp and Sm:
#spin_current_op = 1j* ( np.kron( np.kron(Sp, np.eye(d)) , np.kron(Sm, np.eye(d))) - np.kron( np.kron(Sm, np.eye(d)) , np.kron(Sp, np.eye(d))) )


#### NORM_state initialization
NORM_state = create_maxmixed_normstate()
NORM_state.singlesite_thetas = calculate_thetas_singlesite(NORM_state)
NORM_state.twosite_thetas = calculate_thetas_twosite(NORM_state)


#### Loading and saving states
loadstate_folder = "data\\"
loadstate_filename = "0102_2106_DENS1_N21_chi35.pkl"

save_state_bool = False
load_state_bool = False

####################################################################################
#TimeEvol_obj_half = Time_Operator(N, d, JXY, JZ, h, s_coup, dt/2, Diss_bool, True, use_CN) #first and last half timesteps for even sites


def main():
    #load state or create a new one
    if load_state_bool:
        DENS1 = load_state(loadstate_folder, loadstate_filename, 1)
    else:
        MPS1 = MPS(1, N,d,chi, False)
        #MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_halfstate(N,d,chi)
        MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_LU_RD(N,d,chi, scale_factor = 0.8 )        
        DENS1 = create_superket(MPS1, newchi)
    
    #creating time evolution object
    TimeEvol_obj1 = Time_Operator(N, d, JXY, JZ, h, s_coup, dt, Diss_bool, True, use_CN)
    TimeEvol_obj2 = Time_Operator(N, d, JXY, JZ, h, s_coup, dt, False, False, use_CN)
    
    #declaring which desired operator expectations must be tracked -- not used here
    desired_expectations = []
    #desired_expectations.append(("Sz", np.kron(Sz, np.eye(d)), False, 0))
    pure_desired_expectations = []
    #pure_desired_expectations.append(("Sz", Sz, False, 0))
    
    #time evolution of the state        
    DENS1.time_evolution(TimeEvol_obj1, normalize, steps, desired_expectations, True, False, True)
    MPS1.time_evolution(TimeEvol_obj2, normalize, steps, pure_desired_expectations, True, False, False)
     

    if save_state_bool:
        DENS1.store()

    #Plot sinal spin profile of DENS1
    final_Sz = np.zeros(N)
    for i in range(N):
        final_Sz[i] = DENS1.expval(np.kron(Sz, np.eye(d)), i)
    plt.plot(final_Sz, linestyle="", marker=".")
    plt.xlabel("Site")
    plt.ylabel("<Sz>")
    plt.grid()
    plt.title(f"<Sz> for each site after {steps} steps with dt={dt}")
    plt.show()  
    pass

main()

elapsed_time = time.time()-t0
print(f"Elapsed simulation time: {elapsed_time}")