import numpy as np
from statistics import NormalDist
from math import exp
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import ImageTk, Image
from tkinter import *
from tkinter import ttk,messagebox
from define import *
from pricing import *
from greeks import Greeks
from time import sleep

matplotlib.use('TkAgg')

def plot_greeks():
	curr_tab = NB.index("current")
	if greeks_figures[curr_tab] != []:
		global current_greek
		if current_greek == 4:
			current_greek = 0
		else:
			current_greek += 1
		
		greek_chart = FigureCanvasTkAgg(greeks_figures[curr_tab][current_greek], master=tabs[curr_tab])
		greek_chart.get_tk_widget().grid(row=7, column=4, columnspan=1, rowspan=6)
		greek_chart.draw()

def write(*args):
	curr_tab = NB.index("current")
	pricing_function = [black_scholes_model, binomial_model, jump_diffusion_model, heston_model][curr_tab]
	
	# try-except block to catch if the user clicks the "calculate" button
	# without having correctly inserted all the inputs
	user_inputs = []
	j = 0
	if curr_tab == 1: j += 1
	converter = [float for i in range(n_inputs[curr_tab] - (1 + j))] + [lambda x: x for i in range(1 + j)]
	try:
		for i in range(n_inputs[curr_tab]):
			user_inputs.append(converter[i](tabs_elements[curr_tab][i].get()))
	except ValueError:
		messagebox.showerror("Error","Please check all inputs are correctly inserted")
		return
	user_inputs[T]=user_inputs[T]/365
	user_inputs[t]=user_inputs[t]/365
	tabs_elements[curr_tab][-1].delete(0, END)
	tabs_elements[curr_tab][-1].insert(0, pricing_function(*user_inputs))

	S_a = np.linspace(user_inputs[s] * 0.8, user_inputs[s] * 1.2, 21)
	K_a = np.full(21, user_inputs[k])
	T_a = np.full(21, user_inputs[T])
	t_a = np.full(21, user_inputs[t])
	q_a = np.full(21, user_inputs[q])
	r_a = np.full(21, user_inputs[r])
	sigma_a = np.full(21, user_inputs[sig])
	P_a = np.array([pricing_function(a, b, c, d, e, f, g, *user_inputs[7:]) for a, b, c, d, e, f, g in zip(S_a, K_a, T_a, t_a, q_a, r_a, sigma_a)])
	mat = np.vstack((S_a, K_a, T_a, t_a, q_a, r_a, sigma_a, P_a))

	if user_inputs[-1] == "Call":
		intrinsic_value = np.array([np.maximum(0, s - k) for s, k in zip(S_a, K_a)])
	elif user_inputs[-1] == "Put":
		intrinsic_value = np.array([np.maximum(0, k - s) for s, k in zip(S_a, K_a)])

#==============================================================
	if curr_tab < 2:   
		if curr_tab == 0:
			tp = "E"
		else:
			tp = user_inputs[-2][0]
		
		g = Greeks(*user_inputs[:7])
		delta, delta_S0 = g.Delta(tp, user_inputs[-1])
		gamma, gamma_S0 = g.Gamma(tp, user_inputs[-1])
		rho, rho_S0 = g.Rho(tp, user_inputs[-1])
		vega, vega_S0 = g.Vega(tp, user_inputs[-1])
		theta, theta_S0 = g.Theta(tp, user_inputs[-1])
		greek_values=[delta_S0,gamma_S0,rho_S0,vega_S0,theta_S0]

		figure = Figure(figsize=(6,11), dpi=70)
		a = figure.add_subplot(111)	
		a.plot(S_a, P_a, color="blue", linestyle="-")
		a.scatter(S_a, P_a, color="blue")
		a.plot(S_a, intrinsic_value, color="red", linestyle="-")
		a.scatter(S_a, intrinsic_value, color="red")
		a.set_title ("Option value", fontsize=16)
		a.set_ylabel("Value", fontsize=14)
		a.set_xlabel("$S_0$", fontsize=14)
		chart_type = FigureCanvasTkAgg(figure, master=tabs[curr_tab])
		chart_type.get_tk_widget().grid(row=0, column=4, columnspan=1, rowspan=6)
		chart_type.draw()
		
		rw=1
		for label,greek_value in zip(Greeks_S0,greek_values):
			l = Label(master=tabs[curr_tab], text=label)
			l.grid(row=rw, column=2, columnspan=1)
			f = Entry(master=tabs[curr_tab])
			f.grid(row=rw, column=3, columnspan=1)
			f.insert(0, greek_value)
			rw +=1

		bt = Button(master=tabs[curr_tab], text="Plot next Greek", command=plot_greeks)
		bt.grid(row=7,column=2,columnspan=2)
		
		global greeks_figures
		greeks_figures[curr_tab] = [delta, gamma, rho, vega, theta]
		global current_greek
		current_greek = -1
		
		plot_greeks()
	
#create window=============================================================================
root = Tk()
root.title("Option pricer")
root.geometry("1000x800")
#create tabs
NB = ttk.Notebook(root)
t1 = ttk.Frame(NB)
t2 = ttk.Frame(NB)
t3 = ttk.Frame(NB)
t4 = ttk.Frame(NB)

NB.add(t1, text='Black Scholes model')
NB.add(t2, text='Binomial model')
NB.add(t3, text='Jump diffusion model')
NB.add(t4, text='Heston model')
NB.pack(expand=1, fill="both")

tabs = [t1, t2, t3, t4]
tabs_elements = []
frames = []

#setting each window characteristics\
for tab,n_column in zip(tabs,n_columns):
	tab.rowconfigure([0,1,2,3,4,5,6,7,8,9,10,11,12,13],minsize=30,weight=1)
	tab.columnconfigure(n_column,minsize=30,weight=1)
	
	elements = []
	rw = 1
	for label in labels[:-1]:
		l = Label(master=tab, text=label)
		l.grid(row=rw, column=0, columnspan=1)
		elements.append(Entry(master=tab))
		elements[-1].grid(row=rw, column=1, columnspan=1)
		rw += 1
	
	# we are not in the heston model then volatility must be provided by the user
	if tab != t4:
		l = Label(master=tab, text=labels[-1])
		l.grid(row=rw, column=0, columnspan=1)
		elements.append(Entry(master=tab))
		elements[-1].grid(row=rw, column=1, columnspan=1)
	
	# creating button to compute the price and associate command to start function write
	bt = Button(master=tab, text="Calculate", command=write)
	bt.grid(row=11, column=1,columnspan=1)

	if tab == t2:
		l = Label(master=tab, text="number of nodes")
		l.grid(row=9, column=0, columnspan=1)
		elements.append(Entry(master=tab))
		elements[-1].grid(row=9, column=1, columnspan=1)
		
		# Combobox for type of option
		l = Label(master=tab, text="Type")
		l.grid(row=10, column=0, columnspan=1)
		n = StringVar()
	
		elements.append(ttk.Combobox(tab, width=17, textvariable=n))
		elements[-1]['values'] = option_type
		elements[-1].grid(row=10, column=1)
	else:
		to_use = []
		if tab == t3:
			to_use = j_labels
		elif tab == t4:
			
			to_use = h_labels
		rw = 1
		for label in to_use:
			l = Label(master=tab, text=label)
			l.grid(row=rw, column=2, columnspan=1)
			elements.append(Entry(master=tab))
			elements[-1].grid(row=rw, column=3, columnspan=1)
			rw += 1
		
	# Combobox for call / put choice
	l = Label(master=tab, text="Contract")
	l.grid(row=8, column=0, columnspan=1)
	n = StringVar()

	elements.append(ttk.Combobox(tab, width=17, textvariable=n))
	elements[-1]['values'] = option_list
	elements[-1].grid(row=8, column=1)

	# Entry to see the final price of the option
	elements.append(Entry(master=tab))
	elements[-1].grid(row=12, column=1,columnspan=1)
	l = Label(master=tab,text="Option price")
	l.grid(row=12,column=0,columnspan=1)

	# filling tab elements list
	tabs_elements.append(elements)
	
#=============================================================================================

# greeks graphs variables
greeks_figures = {0 : [], 1 : []}
current_greek = -1

#telling the window to wait for user input
root.mainloop()
