# BEP

This repository contains the Python code for the my bachelor project. It is based on previous work of Matthijs Ates, with changes to the implementation of the Lindblad term.

The final_model is the "user-friendly" code.

The code simulates a XXZ-Heisenberg model between two leads. We use MPS, or vectorised MPDO, as an efficient method. 

It is possible to measure multiple quantities, such as trace, magnetisation and current.

The model works for second-order and third-order numerical time-integration.

The packages os, numpy, scipy.linalg, matplotlib.pyplot, pickle, time, datetime, math, itertools and multiprocessing are required.

The code initializations_mps is used for initialisations of the MPS is used in the main code.

model_1 contains more features, but is not very user-friendly. 






