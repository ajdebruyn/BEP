# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:04:08 2024

@author: sande
"""
# -*- coding: utf-8 -*-

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from scipy.linalg import expm

import matplotlib.pyplot as plt

import pickle
import time
from datetime import datetime
import math

import pandas as pd

from initializations_mps import initialize_halfstate, initialize_LU_RD


########################################################################################################

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
      
    
    def construct_vidal_supermatrices(self, chi):
        """ Constructs a superket of the density operator in Vidal decomposition """
        sup_Gamma_mat = np.zeros((self.N, self.d**2, chi, chi), dtype=complex)
        sup_Lambda_mat = np.zeros((self.N+1, chi))
        for i in range(self.N):
            sup_Gamma_mat[i,:,:,:] = np.kron(self.Gamma_mat[i], np.conj(self.Gamma_mat[i]))[:,:chi,:chi]
            sup_Lambda_mat[i,:] = np.kron(self.Lambda_mat[i], self.Lambda_mat[i])[:chi]
        sup_Lambda_mat[N,:] = np.kron(self.Lambda_mat[N], self.Lambda_mat[N])[:chi]
        sup_locsize = np.minimum(self.locsize**2, chi)
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
        # print(np.shape(theta), np.shape(TimeOp))
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
    
    def TEBD(self, TimeOp, Lindbladian_arr, normalize, step):
        """ TEBD algorithm, L incl H at both ends """
        
        if step == 0:
            for i in range(2, self.N-2, 2):                         # H_o /2
                self.apply_twosite(TimeOp[i]/2, i, normalize)
        else:
            for i in range(2, self.N-2, 2):                         # H_o
                self.apply_twosite(TimeOp[i], i, normalize)
        
        for i in range(3, self.N-3, 2):                         # H_e 
            self.apply_twosite(TimeOp[i], i, normalize)
        self.apply_twosite(TimeOp[1]/2, 1, normalize)           # H_e /2 at link 1 
        if self.N%2 == 0:                                            # and link N-3 IF overlaps with L
            self.apply_twosite(TimeOp[self.N-3]/2, self.N-3, normalize)
                  
        for i in range(len(Lindbladian_arr["index"])):          # Lindblad at links 0 and N-2
            self.apply_twosite(Lindbladian_arr["TimeOp"][i], Lindbladian_arr["index"][i], normalize)
        
        self.apply_twosite(TimeOp[1]/2, 1, normalize)           # H_e /2 at link 1 
        if self.N%2 == 0:                                            # and link N-3 IF overlaps with L
            self.apply_twosite(TimeOp[self.N-3], self.N-3, normalize)
        
        if step == steps-1:
            for i in range(2, self.N-2, 2):                         # H_o /2
                self.apply_twosite(TimeOp[i]/2, i, normalize)
        
        return
        
    def expval(self, Op, site):
        """ Calculates the expectation value of an operator Op for a single site """
        Gamma_temp = self.Gamma_mat[site].copy()
        self.apply_singlesite(Op, site)
        result = self.calculate_vidal_inner(NORM_state)
        self.Gamma_mat[site] = Gamma_temp
        return result
    
    def expval_chain(self, Op):
        """ calculates expectation value for operator Op for the entire chain """
        result = np.zeros(self.N)
        for i in range(self.N):
            result[i] = self.expval(Op, i)
        return result
        
    def expval_twosite(self, Op, site):
        """ Calculates expectation value for a twosite operator Op at sites site and site+1 """
        temp_Lambda = self.Lambda_mat[site+1].copy()
        temp_Gamma = self.Gamma_mat[site:site+2].copy()
        Op = Op.reshape(self.d,self.d,self.d,self.d)
        self.apply_twosite(Op,site,False)
        result = self.calculate_vidal_inner(NORM_state)
        self.Lambda_mat[site+1] = temp_Lambda
        self.Gamma_mat[site:site+2] = temp_Gamma
        return result
   
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
        if track_normalization==False:
            print("Tracking normalization is required for calculating expectation values in DENS foralism")
            print("Be aware this may cause issues")
            
        
        #### Initializing operators and expectation value arrays
        TimeOp = TimeEvol_obj.TimeOp
        # Diss_arr = TimeEvol_obj.Diss_arr
        TimeOp_Lindblad = TimeEvol_obj.TimeOp_Lindblad
        
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
                self.trace = np.append(self.trace, self.calculate_vidal_inner(NORM_state))
            if track_energy:
                energy[t] = self.calculate_energy(TimeEvol_obj) /self.trace[t]
            if track_current:
               middle_site = int(np.round(self.N/2-1))
               self.spin_current_values = np.append(self.spin_current_values, np.real( self.expval_twosite(spin_current_op, middle_site) ))
               self.spin_current_values[-1] *= 1/self.trace[t]
            
            for i in range(self.N):
                Sz_expvals[i,t] = self.expval(np.kron(Sz, np.eye(d)), i)
            Sz_expvals[:,t] *= 1/self.trace[t]
                
            self.TEBD(TimeOp, TimeOp_Lindblad, normalize, t)
        
        
        data["Norm_end"][asdf,jkl] = (self.normalization)[-1]
        data["Norm_high"][asdf,jkl] = max(abs(1-self.normalization))
        data["Trace_end"][asdf,jkl] = (self.trace)[-1]
        data["Trace_high"][asdf,jkl] = max(abs(1-self.trace))
        data["Trace_max"][asdf,jkl] = max(self.trace[:int(2*steps/t_end)])
        data["current_end"][asdf,jkl] = (self.spin_current_values)[-1]
 
        #### Plotting expectation values
        
        time_axis = np.arange(steps)*abs(TimeEvol_obj.dt)
        
        osc_max = 0
        
        for i in range(self.N):
            plt.plot(time_axis, Sz_expvals[i,:], label="Site "+str(i+1))
            osc_max = max(osc_max, max(Sz_expvals[i,-int(steps/t_end):]) - min(Sz_expvals[i,-int(steps/t_end):]))
  #      plt.title("Expectation value of $S_Z$  over time")
        plt.xlabel("Time $t$")
        plt.ylabel("$〈 S_Z 〉$")
        plt.grid()
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.savefig('sol_sz.png', dpi=500, bbox_inches='tight')
        plt.show()
        
        data["Sz_osc"][asdf,jkl] = osc_max
        
        if track_normalization:
            plt.plot(time_axis, self.normalization[-steps:])
     #       plt.title("Normalization over time")
            plt.xlabel("Time $t$")
            plt.ylabel("Norm of $|ρ〉〉$")
            plt.grid()
            plt.savefig('sol_norm.png', dpi=500, bbox_inches='tight')
            plt.show()
            
            plt.plot(time_axis, self.trace[-steps:])
        #    plt.title("Trace over time")
            plt.xlabel("Time $t$")
            plt.ylabel("Trace of $|ρ〉〉$")
            plt.grid()
            plt.savefig('sol_trace.png', dpi=500, bbox_inches='tight')
            plt.show()
        
        if track_energy:
            plt.plot(time_axis, energy)
        #    plt.title("Energy over time")
            plt.xlabel("Time $t$")
            plt.ylabel("Energy $E$")
            plt.grid()
            plt.savefig('sol_energy.png', dpi=500, bbox_inches='tight')
            plt.show()
        
        if (track_current==True):
            plt.plot(time_axis, self.spin_current_values)
           # plt.title("Current over time")
            plt.xlabel("Time $t$")
            plt.ylabel("Current $I$")
            plt.grid()
            plt.savefig('sol_current.png', dpi=500, bbox_inches='tight')
            plt.show()
            
        return
    

########################################################################################  

class Time_Operator:
    def __init__(self, N, d, JXY, JZ, h, s_coup, weightLR, dt):
        self.N = N
        self.d = d
        self.JXY = JXY
        self.JZ = JZ
        self.h = h
        self.s_coup = s_coup
        self.dt = dt
        self.weightLR = weightLR
        self.order = 2
        
        if isinstance(dt, complex):     
            self.is_imaginary = True
        else:
            self.is_imaginary = False
       
        #### Creating Hamiltonian and Time operators
        #### Note: Ham_energy is the Hamiltonian to be used for energy calculation
        self.Ham, self.Ham_energy, self.Ham_left, self.Ham_right = self.Create_Dens_Ham()
        
        self.TimeOp = self.Create_TimeOp(self.dt)
        
        self.TimeOp_Lindblad = self.create_TimeOp_Lindblad(self.weightLR, self.dt, self.order)
        
        return
        
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
       
            H_arr[i, 1:self.N-2,:,:] = H_M
            H_arr[i, 0,:,:] = H_L
            H_arr[i, self.N-2,:,:] = H_R

        #Note: H_arr[0] is the correct Hamiltonian to use for energy calculations
        return (H_arr[0] - np.conj(H_arr[1])), H_arr[0], H_arr[0], H_arr[1]     

    def Create_TimeOp(self, dt):
        ''' Creates the Hamiltonian time operator '''
        U = np.ones((self.N-1, self.d**4, self.d**4), dtype=complex)
        
        U[0,:,:] = expm(-1j*dt*self.Ham[0])
        U[self.N-2,:,:] = expm(-1j*dt*self.Ham[self.N-2])
        U[1:self.N-2,:,:] = expm(-1j*dt*self.Ham[1]) # we use broadcasting
    
        U = np.around(U, decimals=15)        #Rounding out very low decimals 
        return np.reshape(U, (self.N-1,self.d**2,self.d**2,self.d**2,self.d**2))
    
    def Calculate_J(self, Lind_Op, site):
        """ Calculates J as in Cao Eq. (5), note that due to vectorization  """
        J_left = -1j * self.Ham_left[site]
        J_right = 1j * self.Ham_right[site]
        for i in range(np.shape(Lind_Op)[0]):
            if site == 0:       # First site
                J_right -= np.kron( np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op[i]), np.conj(Lind_Op[i]))), np.eye(self.d**2) ) /2
                J_left -= np.kron( np.kron(np.matmul(np.conj(np.transpose(Lind_Op[i])), Lind_Op[i]), np.eye(self.d)), np.eye(self.d**2) ) /2
            else:               # Last site
                J_right -= np.kron( np.eye(self.d**2), np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op[i]), np.conj(Lind_Op[i]))) ) /2
                J_left -= np.kron( np.eye(self.d**2), np.kron(np.matmul(np.conj(np.transpose(Lind_Op[i])), Lind_Op[i]), np.eye(self.d)) ) /2
        return J_left, J_right
    
    def create_J_m(self, Lind_Op, site, ts, m):
        """ Approximate exp(L_J (t-s)) by J_m(t-s) as in Cao Eq. (11) """
        temp_L = np.eye(self.d**4)
        temp_R = np.eye(self.d**4)
        for i in range(1,m+1):
            temp_L = np.add(temp_L, self.Calculate_J(Lind_Op, site)[0]**i * abs(ts)**i / math.factorial(i))
            temp_R = np.add(temp_R, self.Calculate_J(Lind_Op, site)[1]**i * abs(ts)**i / math.factorial(i))
        return np.matmul(temp_L, temp_R)
    
    def Create_F(self, Lind_Op, site, order, m, s_val):
        """ Create F_m^M as in Cao Eq. (13) """
        temp = self.create_J_m(Lind_Op, site, s_val[0], m)
        for i in range(order):
            temp2 = np.matmul(self.create_J_m(Lind_Op, site, (s_val[i+1]-s_val[i]), m), self.Calculate_LL_site(Lind_Op, site))
            temp = np.matmul(temp2, temp)
        return temp
    
    def Create_Duhamel(self, Lind_Op, site, order):
        """ Create approximation of the Duhamel notation up to 2nd order as in Cao Eq. (10) """
        return self.create_J_m(Lind_Op, site, order, 0, self.dt) + self.dt * self.create_J_m(Lind_Op, site, order-1, 0, self.dt) * self.Approximate_LL(Lind_Op, site, order)
    
    def Lindblad_operators(self, weights):
        """ Create the set of Lindblad operators including their weights """
        return weights[0]*np.eye(d), weights[1]*Sp, weights[2]*Sm, weights[3]*Sz    

    def Calculate_LJ_site(self, Lind_Op, site):
        """ Creates the L_J term for a double site """
        """ Lind_Op is shape (k,d,d) -- the k-index is for multiple different lindblad operators that act on a single site """
        temp = np.zeros((self.d**2, self.d**2), dtype=complex)
        for i in range(np.shape(Lind_Op)[0]):
            temp -= np.kron(np.matmul(np.conj(np.transpose(Lind_Op[i])), Lind_Op[i]), np.eye(self.d))/2
            temp -= np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op[i]), np.conj(Lind_Op[i])))/2
        if site == 0:
            LJ = np.kron(temp, np.eye(self.d**2))
        else:
            LJ = np.kron(np.eye(self.d**2), temp)
        return LJ - 1j*self.Ham[site]
    
    def Calculate_LL_site(self, Lind_Op, site):
        """ Creates the L_L term for a double site """
        """ Lind_Op is shape (k,d,d) -- the k-index is for multiple different lindblad operators that act on a single site """
        temp = np.zeros((self.d**2, self.d**2), dtype=complex)              
        for i in range(np.shape(Lind_Op)[0]):
            temp += np.kron(Lind_Op[i], np.conj(Lind_Op[i]))
        if site == 0:
            return np.kron(temp, np.eye(self.d**2))
        else:
            return np.kron(np.eye(self.d**2), temp)
    
    def Approximate_LL(self, Lind_Op, site, dt, order):
        """ Approximates exp(dt L_L) by a Taylor series truncation """
        LL = self.Calculate_LL_site(Lind_Op, site)
        temp = np.eye(self.d**4)
        for j in range(1, order+100):
            temp = np.add(temp, LL**j * dt**j / math.factorial(j))
        return temp
    
    def Approximate_LJ(self, Lind_Op, site, dt, order): 
        """ Approximates exp(dt L_J) by a 'Taylor series' truncation in Kraus form """
        LJ = self.Calculate_LJ_site(Lind_Op, site)
        temp = np.eye(self.d**4)
        for j in range(1,order+100):
            temp = np.add(temp, LJ**j * dt**j / math.factorial(j))
        return temp
    
    def Calculate_L_site(self, Lind_Op, site, dt, order):
        ''' Calculates the total exp(dt L), second order '''
        temp = self.Approximate_LL(Lind_Op, site, dt/2, order)
        temp2 = np.matmul(temp, np.matmul(self.create_J_m(Lind_Op, site, dt, order), temp))
        return np.reshape(temp2, (self.d**2, self.d**2, self.d**2, self.d**2))
    
    def create_TimeOp_Lindblad(self, weightLR, dt, order):
        ''' Creates the Time operator exp(dt L) '''
        Lb_arr = np.zeros((), dtype=[
            ("index", int, 2),
            ("Operator", complex, (2, self.d**4, self.d**4)),
            ("TimeOp", complex, (2, self.d**2, self.d**2, self.d**2, self.d**2))
            ])
        Lb_arr["index"][0] = 0
        Lb_arr["index"][1] = N-2
        
        Lind_Op = np.zeros((self.d,self.d**2,self.d,self.d))
        for i in range(len(Lb_arr["index"])):
            Lind_Op[i] = self.Lindblad_operators(weightLR[i])
            Lb_arr["Operator"][i,:,:] = self.Calculate_LJ_site(Lind_Op[i], Lb_arr["index"][i]) + self.Calculate_LL_site(Lind_Op[i], Lb_arr["index"][i])
            Lb_arr["TimeOp"][i,:,:,:,:] = self.Calculate_L_site(Lind_Op[i], Lb_arr["index"][i], dt, order)        
        return Lb_arr



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
    

def create_superket(State, chi):
    """ create MPS of the density matrix of a given MPS """
    gammas, lambdas, locsize = State.construct_vidal_supermatrices(chi)
    
    name = "DENS" + str(State.ID)
    newDENS = MPS(State.ID, State.N, State.d**2, chi, True)
    newDENS.Gamma_mat = gammas
    newDENS.Lambda_mat = lambdas
    newDENS.locsize = locsize
    globals()[name] = newDENS
    return newDENS

def create_maxmixed_normstate():
    """ Creates vectorised density matrix of an unnormalized maximally mixed state, used to calculate the trace of a vectorised density matrix """
    """ since to obtain rho11 + rho22 you must take inner [1 0 0 1] [rho11 rho12 rho21 rho22]^T without a factor 1/sqrt(2) in front """
    lambdas = np.zeros((N+1,chi))
    lambdas[:,0]= 1
    
    gammas = np.zeros((N,d**2,chi,chi), dtype=complex)
    diagonal = (1+d)*np.arange(d)
    gammas[:,diagonal, 0, 0] = 1        #/2  #/np.sqrt(2)
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi**2)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum((d**2)**arr, chi**2)
    
    NORM_state = MPS(0, N, d**2, chi, True)
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
N           = 12         #System Length
d           = 2         #Spin dimension -- do not change
#chi         = 20        #Dens truncation parameter

im_steps    = 0         #Number of timesteps in imaginary time
im_dt       = -0.03j    #Size of imaginary timestep
#steps       = 400      # '' for real timestep
#dt          = 0.05      # '' for real timestep

normalize   = True      #whether to maintain MPS normalization

#### Hamiltonian and Lindblad constants
h           = 0         #Strength of the transverse magnetic field
JXY         = 1         #XY Coupling strength XXZ Hamiltonian
JZ          = 1         #Z coupling strength of XXZ Hamiltonian

s_coup      = 1         #Coupling strength of Dissipative terms
s_coup      = np.sqrt(s_coup)  
weightLR    = [[0,2,0,0],[0,0,2,0]]

# [[0,1,0,0],[0,0,1,0]] [[1,1,1,1],[1,1,1,1]]


#### Spin matrices
Sp          = np.array([[0,1],[0,0]])
Sm          = np.array([[0,0],[1,0]])
Sx          = np.array([[0,1], [1,0]])
Sy          = np.array([[0,-1j], [1j,0]])
Sz          = np.array([[1,0],[0,-1]])

t_end = 40


#### Spin current operator and cutoff factor
spin_current_op = 2 * ( np.kron( np.kron(Sx, np.eye(d)) , np.kron(Sy, np.eye(d))) - np.kron( np.kron(Sy, np.eye(d)) , np.kron(Sx, np.eye(d))) )
#equivalent operator in terms of Sp and Sm:
#spin_current_op = 1j* ( np.kron( np.kron(Sp, np.eye(d)) , np.kron(Sm, np.eye(d))) - np.kron( np.kron(Sm, np.eye(d)) , np.kron(Sp, np.eye(d))) )


#### NORM_state initialization


#### Loading and saving states
loadstate_folder    = "data\\"
loadstate_filename  = "0102_2106_DENS1_N21_chi35.pkl"

save_state_bool     = False
load_state_bool     = False

####################################################################################
#TimeEvol_obj_half = Time_Operator(N, d, JXY, JZ, h, s_coup, dt/2, True) #first and last half timesteps for even sites


def main():
    #load state or create a new one
    if load_state_bool:
        DENS1 = load_state(loadstate_folder, loadstate_filename, 1)
    else:
        MPS1 = MPS(1, N,d,chi, False)
        #MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_halfstate(N,d,chi)
        MPS1.Gamma_mat[:,:,:,:], MPS1.Lambda_mat[:,:], MPS1.locsize[:] = initialize_LU_RD(N,d,chi, scale_factor = 0.8 )        
        DENS1 = create_superket(MPS1, chi)
    
    #creating time evolution object
    TimeEvol_obj1 = Time_Operator(N, d, JXY, JZ, h, s_coup, weightLR, dt)
    #TimeEvol_obj2 = Time_Operator(N, d, JXY, JZ, h, s_coup, dt, False, False)
    
    #declaring which desired operator expectations must be tracked -- not used here
    desired_expectations = []
    #desired_expectations.append(("Sz", np.kron(Sz, np.eye(d)), False, 0))
    #pure_desired_expectations = []
    #pure_desired_expectations.append(("Sz", Sz, False, 0))
    
    #time evolution of the state        
    DENS1.time_evolution(TimeEvol_obj1, normalize, steps, desired_expectations, True, True, True)
    #MPS1.time_evolution(TimeEvol_obj2, normalize, steps, pure_desired_expectations, True, False, False)
     

    if save_state_bool:
        DENS1.store()

    #Plot sinal spin profile of DENS1
    final_Sz = np.zeros(N)
    for i in range(N):
        final_Sz[i] = DENS1.expval(np.kron(Sz, np.eye(d)), i)
        data["Sz"][i,asdf,jkl] = final_Sz[i]
    plt.plot(range(1,N+1), final_Sz, linestyle="", marker=".")
    plt.xticks(range(1,N+1))
    plt.xlabel("Site")
    plt.ylabel("$〈 S_Z 〉$")
    plt.grid()
#    plt.title(f"Expectation value of $〈 S_Z 〉$ for each site") #after {steps+1} steps with dt={dt}")
    plt.savefig('sol_sz_site.png', dpi=500, bbox_inches='tight')
    plt.show()  
    pass


t_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.085, 0.09, 0.095]
chi_range = [9, 11, 13, 15, 17]

data = np.zeros((), dtype=[
    ("t_range", complex, (len(t_range))),
    ("chi_range", complex, (len(chi_range))),
    ("Sz", complex, (N, len(t_range), len(chi_range))),
    ("Sz_osc", complex, (len(t_range), len(chi_range))),
    ("Norm_end", complex, (len(t_range), len(chi_range))),
    ("Norm_high", complex, (len(t_range), len(chi_range))),
    ("Trace_end", complex, (len(t_range), len(chi_range))),
    ("Trace_high", complex, (len(t_range), len(chi_range))),
    ("Trace_max", complex, (len(t_range), len(chi_range))),
    ("current_end", complex, (len(t_range), len(chi_range)))
    ])

data["t_range"] = t_range[0]
data["chi_range"] = chi_range[0]

timer = time.time()

for asdf in range(len(t_range)):
    dt = t_range[asdf]
    for jkl in range(len( chi_range)):
        chi = chi_range[jkl]
        steps = round(t_end/dt)
        
        NORM_state                      = create_maxmixed_normstate()
        NORM_state.singlesite_thetas    = calculate_thetas_singlesite(NORM_state)
        NORM_state.twosite_thetas       = calculate_thetas_twosite(NORM_state)
        
        print(asdf,jkl)
        
        main()
        
        filename = 'C:\\Users\\sande\\Downloads\\BEPtest1'
        np.save(filename, data)
        
        print(asdf,jkl)
        
        if time.time()-timer > 10*60:
            if time.time()-timer > 20*60:
                time.sleep(200)
            else:
                time.sleep(100)
            timer = time.time()
        else:
            pass
        





elapsed_time = time.time()-t0
sec_round = round(elapsed_time)
print(f"Elapsed simulation time: {sec_round} seconds")

if math.floor(elapsed_time/3600)>0:
    hours = math.floor(elapsed_time/3600)
    minutes = math.floor((elapsed_time-3600*hours)/60)
    seconds = round(elapsed_time-3600*hours-60*minutes)
    print(f"Elapsed simulation time: {hours}h {minutes}m {seconds}s")
elif math.floor(elapsed_time/60)>0:
    minutes = math.floor((elapsed_time/60))
    seconds = round(elapsed_time-60*minutes)
    print(f"Elapsed simulation time: {minutes}m {seconds}s")











