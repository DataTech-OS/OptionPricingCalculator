# This library just contains enumeration definitions
# in order to make the main code more readable

s = 0
k = 1
T = 2
t = 3
q = 4
r = 5
sig = 6
cont_tp = 7
n_nodes = 8
usa = 9
price = 10

n_inputs = [8, 10, 11, 12]
labels = ["S\u2080", "K", "T", "t", "q", "r", "sigma"]
h_labels = ["kappa", "theta", "vvol", "rho", "V\u2080"]
j_labels = ["lambda", "a", "b\u00B2"]
option_list = ("Call", "Put")
option_type = ("European", "American")
n_columns=[[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3],[0,1,2,3]]
Greeks_S0=["Delta","Gamma","Rho","Vega","Theta"]