# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:38:50 2024

@author: sande
"""
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:03:51 2024

@author: sande
"""
import numpy as np
import matplotlib.pyplot as plt


data = np.load('C:\\Users\\sande\\Downloads\\BEP_cao3_dt.npy')

eind = len(data["t_range"])

name = 'cao'

t_range =  [num.real for num in data["t_range"]][0:eind]
chi_range = [int(num.real) for num in data["chi_range"]]
#t_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.085, 0.09, 0.095]
#chi_range = [9, 11, 13, 15, 17]


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


for i in range(len(np.transpose(current))):
    plt.plot(t_range, np.transpose(current)[i][0:eind], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
#plt.title("Current at $t$ = 25")
plt.xlabel(r"Timestep $\Delta t$")
plt.ylabel("Current $I$")
plt.grid()
plt.legend(loc=2)
plt.savefig(f'{name}_dt_current.png', dpi=500)
plt.show()



for i in range(len(np.transpose(Sz_osc))):
    plt.plot(t_range, np.transpose(Sz_osc)[i][0:eind], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
#plt.title("Amplitude of oscillations")
plt.xlabel(r"Timestep $\Delta t$")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.savefig(f'{name}_dt_oscillations.png', dpi=500)
plt.show()

for i in range(len(np.transpose(Sz_osc))):
    plt.loglog(t_range, abs(np.transpose(Sz_osc)[i][0:eind]-0.0015), label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
plt.loglog(t_range, 3.7*np.array(t_range)**2, label = "Quadratic fit with offset", linestyle='--')
#plt.title("Amplitude of oscillations")
plt.xlabel(r"log($\Delta t$)")
plt.ylabel("log(Amplitude)")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig(f'{name}_dt_loglog_oscillations.png', dpi=500)
plt.show()



for i in range(len(np.transpose(norm_end))):
    plt.plot(t_range, np.transpose(norm_end)[i][0:eind], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
#plt.title("Norm at $t$ = 25")
plt.xlabel(r"Timestep $\Delta t$")
plt.ylabel("Norm")
plt.grid()
plt.legend()
plt.savefig(f'{name}_dt_norm_end.png', dpi=500, bbox_inches='tight')
plt.show()

for i in range(len(np.transpose(norm_end))):
    plt.loglog(t_range, np.transpose(norm_end)[i][0:eind]-1, label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
plt.loglog(t_range, 0.072*np.array(t_range)**2.1, label = r"$\Delta t^{2.5}$ fit", linestyle='--')
#plt.title("Norm at $t$ = 25")
plt.xlabel(r"log($\Delta t$)")
plt.ylabel("log(Norm error)")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig(f'{name}_dt_loglog_norm_end.png', dpi=500, bbox_inches='tight')
plt.show()



for i in range(len(np.transpose(norm_high))):
    plt.plot(t_range, np.transpose(norm_high)[i][0:eind], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
#plt.title("Maximum norm error/Norm peak")
plt.xlabel(r"Timestep $\Delta t$")
plt.ylabel("Norm error")
plt.grid()
plt.legend()
plt.savefig(f'{name}_dt_norm_error.png', dpi=500, bbox_inches='tight')
plt.show()

for i in range(len(np.transpose(norm_high))):
    plt.loglog(t_range, np.transpose(norm_high)[i][0:eind], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
plt.loglog(t_range, 0.075*np.array(t_range)**2.1, label = r"$\Delta t^{2.5}$ fit", linestyle='--')
#plt.title("Maximum norm error/Norm peak")
plt.xlabel(r"log($\Delta t$)")
plt.ylabel("log(Norm error)")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig(f'{name}_dt_loglog_norm_error.png', dpi=500, bbox_inches='tight')
plt.show()



for i in range(len(np.transpose(trace_end))):
    plt.plot(t_range, np.transpose(trace_end)[i][0:eind], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
#plt.title("Trace at $t$ = 25")
plt.xlabel(r"Timestep $\Delta t$")
plt.ylabel("Trace")
plt.grid()
plt.legend()
plt.savefig(f'{name}_dt_trace_end.png', dpi=500)
plt.show()

# for i in range(len(np.transpose(trace_end))):
#     plt.loglog(t_range, np.transpose(trace_end)[i], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
# #plt.loglog(t_range, 500*np.array(t_range)**2, label = "Quadratic fit", linestyle='--')
# #plt.title("Trace at $t$ = 25")
# plt.xlabel(r"log($\Delta t$)")
# plt.ylabel("log(Trace error)")
# plt.grid(True, which="both", ls="-")
# plt.legend()
# plt.savefig(f'{name}_loglog_trace_end.png', dpi=500)
# plt.show()



for i in range(len(np.transpose(trace_max))):
    plt.plot(t_range, np.transpose(trace_max)[i], label = r"$\chi$ = "+str(chi_range[i]), marker = 's')
plt.title("Trace peak")
plt.xlabel(r"Timestep $\Delta t$")
plt.ylabel("Trace")
plt.grid()
plt.legend()
plt.savefig(f'{name}_dt_trace_peak.png', dpi=500)
plt.show()

for i in range(len(np.transpose(trace_max))):
    plt.loglog(t_range, np.transpose(trace_max)[i]-1, label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
plt.loglog(t_range, 2.3*np.array(t_range)**2.5, label = r"$\Delta t^{2.5}$ fit", linestyle='--')
#plt.title("Trace at $t$ = 25")
plt.xlabel(r"log($\Delta t$)")
plt.ylabel("log(Trace error)")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig(f'{name}_loglog_trace_end.png', dpi=500)
plt.show()


for i in range(len(np.transpose(trace_high))):
    plt.plot(t_range, np.transpose(trace_high)[i][0:eind], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
#plt.title("Maximum trace error")
plt.xlabel(r"Timestep $\Delta t$")
plt.ylabel("Trace error")
plt.grid()
plt.legend()
plt.savefig(f'{name}_dt_trace_error.png', dpi=500)
plt.show()

for i in range(len(np.transpose(trace_high))):
    plt.loglog(t_range, np.transpose(trace_high)[i], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
#plt.title("Maximum trace error")
plt.xlabel(r"log($\Delta t$)")
plt.ylabel("log(Trace error)")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig(f'{name}_loglog_trace_error.png', dpi=500)
plt.show()



for n in range(6):
    for i in range(len(np.transpose(Sz[n]))):
        plt.plot(t_range, np.transpose(Sz[n])[i][0:eind]/np.transpose(trace_end)[i][0:eind], label = r"$\chi$ = "+str(chi_range[i]), marker = "s")
#    plt.title(f"Expectation value of $S_Z$ at site {n+1} at $t$ = 25")
    plt.xlabel(r"Timestep $\Delta t$")
    plt.ylabel("$〈 S_Z 〉$")
    plt.grid()
    if n==0:
        plt.legend()
    plt.savefig(f'{name}_dt_sz_{n+1}.png', dpi=500)
    plt.show()

















    