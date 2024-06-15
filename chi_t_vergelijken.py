# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:03:51 2024

@author: sande
"""
import numpy as np
import matplotlib.pyplot as plt


data = np.load('C:\\Users\\sande\\Downloads\\BEP_cao3_chi.npy')

t_range =  [num.real for num in data["t_range"]]
chi_range = [int(num.real) for num in data["chi_range"]]
#t_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.085, 0.09, 0.095]
#chi_range = [9, 11, 13, 15, 17]

name = 'cao'

Sz = data["Sz"]
Sz_osc = data["Sz_osc"] / 2
current = data["current_end"]
norm_end = data["Norm_end"]
norm_high = data["Norm_high"]
trace_end = data["Trace_end"]
trace_high = data["Trace_high"]
trace_max = data["Trace_max"]

def test(x, a, b):
    return a*(x**b)


for i in range(len((current))):
    plt.plot(chi_range, current[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
#plt.title("Current at $t$ = 25")
plt.xlabel(r"Bond dimension $\chi$")
plt.ylabel("Current $I$")
plt.grid()
plt.legend(loc=1)
plt.savefig(f'{name}_chi_current.png', dpi=500)
plt.show()



for i in range(len((Sz_osc))):
    plt.plot(chi_range, (Sz_osc)[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
#plt.title("Amplitude of oscillations")
plt.xlabel(r"Bond dimension $\chi$")
plt.ylabel("Amplitude")
plt.grid()
plt.legend(loc=1)
plt.savefig(f'{name}_chi_oscillations.png', dpi=500)
plt.show()

# for i in range(len((Sz_osc))):
#     plt.loglog(chi_range, (Sz_osc)[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
# plt.loglog(chi_range, 0.037*np.array(chi_range)**0.5, label = "Square root fit", linestyle='--')
# #plt.title("Amplitude of oscillations")
# plt.xlabel(r"log($\chi$)")
# plt.ylabel("log(Amplitude)")
# plt.grid(True, which="both", ls="-")
# plt.legend()
# plt.savefig('chi_loglog_oscillations.png', dpi=500)
# plt.show()



for i in range(len((norm_end))):
    plt.plot(chi_range, (norm_end)[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
#plt.title("Norm at $t$ = 25")
plt.xlabel(r"Bond dimension $\chi$")
plt.ylabel("Norm")
plt.grid()
plt.legend()
plt.savefig(f'{name}_chi_norm_end.png', dpi=500, bbox_inches='tight')
plt.show()

# for i in range(len((norm_end))):
#     plt.loglog(chi_range, (norm_end)[i]-1, label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
# plt.loglog(chi_range, 0.022*np.array(chi_range)**2, label = "Quadratic fit", linestyle='--')
# #plt.title("Norm at $t$ = 25")
# plt.xlabel(r"log($\chi)")
# plt.ylabel("log(Norm error)")
# plt.grid(True, which="both", ls="-")
# plt.legend()
# plt.savefig('chi_loglog_norm_end.png', dpi=500, bbox_inches='tight')
# plt.show()



for i in range(len((norm_high))):
    plt.plot(chi_range, (norm_high)[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
#plt.title("Maximum norm error/Norm peak")
plt.xlabel(r"Bond dimension $\chi$")
plt.ylabel("Norm error")
plt.grid()
plt.legend()
plt.savefig(f'{name}_chi_mps_norm_error.png', dpi=500, bbox_inches='tight')
plt.show()

# for i in range(len((norm_high))):
#     plt.loglog(chi_range, (norm_high)[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
# plt.loglog(chi_range, 0.055*np.array(chi_range)**2, label = "Quadratic fit", linestyle='--')
# #plt.title("Maximum norm error/Norm peak")
# plt.xlabel(r"log($\chi$)")
# plt.ylabel("log(Norm error)")
# plt.grid(True, which="both", ls="-")
# plt.legend()
# plt.savefig('chi_loglog_norm_error.png', dpi=500, bbox_inches='tight')
# plt.show()



for i in range(len((trace_end))):
    plt.plot(chi_range, (trace_end)[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
#plt.title("Trace at $t$ = 25")
plt.xlabel(r"Bond dimension $\chi$")
plt.ylabel("Trace")
plt.grid()
plt.legend()
plt.savefig(f'{name}_chi_trace_end.png', dpi=500)
plt.show()

# for i in range(len((trace_end))):
#     plt.loglog(chi_range, (trace_end)[i]-1, label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
# #plt.title("Trace at $t$ = 25")
# plt.xlabel(r"log($\chi$)")
# plt.ylabel("log(Trace error)")
# plt.grid(True, which="both", ls="-")
# plt.legend()
# plt.savefig('chi_loglog_trace_end.png', dpi=500)
# plt.show()



# for i in range(len(np.transpose(trace_max))):
#     plt.plot(t_range, np.transpose(trace_max)[i], label = r"$\chi$ = "+str(chi_range[i]))
# plt.title("Trace peak")
# plt.xlabel(r"Timestep $\Delta t$")
# plt.ylabel("Trace")
# plt.grid()
# plt.legend()
# plt.savefig('mps_trace_peak.png', dpi=500)
# plt.show()



for i in range(len((trace_high))):
    plt.plot(chi_range, (trace_high)[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
#plt.title("Maximum trace error")
plt.xlabel(r"Bond dimension $\chi$")
plt.ylabel("Trace error")
plt.grid()
plt.legend()
plt.savefig(f'{name}_chi_trace_error.png', dpi=500)
plt.show()

# for i in range(len((trace_high))):
#     plt.loglog(chi_range, (trace_high)[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
# #plt.title("Maximum trace error")
# plt.xlabel(r"log($\chi$)")
# plt.ylabel("log(Trace error)")
# plt.grid(True, which="both", ls="-")
# plt.legend()
# plt.savefig('chi_loglog_trace_error.png', dpi=500)
# plt.show()



for n in range(6):
    for i in range(len((Sz[n]))):
        plt.plot(chi_range, (Sz[n])[i]/trace_end[i], label = r"$\Delta t$ = "+str(t_range[i]), marker = "s")
#    plt.title(f"Expectation value of $S_Z$ at site {n+1} at $t$ = 25")
    plt.xlabel(r"Bond dimension $\chi$")
    plt.ylabel("$〈 S_Z 〉$")
    plt.grid()
    if n==0:
        plt.legend()
    plt.savefig(f'{name}_chi_sz_{n+1}.png', dpi=500)
    plt.show()

















    