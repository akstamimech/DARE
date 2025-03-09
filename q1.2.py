#Script for numerical solution to optimised DARE

import numpy as np 
import matplotlib.pyplot as plt 


import numpy as np

# System matrices
A = np.array([[0.9, 0.1],
              [0, 0.95]])

B = np.array([[1, 0],
              [0, 1]])

# Cost function matrices
Q = np.array([[5, 0],
              [0, 5]])

R = np.array([[1, 0],
              [0, 1]])



P_init = np.array([[0, 0], [0,0]])


print(Q + np.transpose(A) * P_init * A - (np.transpose(A) @ P_init  @ B /(R + np.transpose(B) @ P_init @ B)) @  np.transpose(B) @  P_init @ A)



P_val = Q + A.T @ P_init @ A - (A.T @ P_init @ B) @ np.linalg.inv((R + B.T @ P_init @ B)) @  B.T @  P_init @ A
currentP = []

currentP.append(P_val)
for n in range(1000): 
    
    currentP.append(Q + A.T @ currentP[n] @ A - (A.T @ currentP[n] @ B) @ np.linalg.inv((R + B.T @ currentP[n] @ B)) @  B.T @  currentP[n] @ A)

print(currentP[999])

P_final = currentP[999]

K = A @ P_final @ B.T @ np.linalg.inv(R + B.T @ P_final @ B)

print(K)

""""
 Now we find the solution for the differential equation of a closed loop system to determine the stability of the system considering given input

"""""

StabM = A - B @ K

print(f'Closed loop system is {StabM} with eigenvalues {np.linalg.eig(StabM)}')