from matplotlib.figure import Figure
from scipy.stats import norm
import numpy as np
from pricing import binomial_model

def plot(x, y, ttl, xl, yl):
    figure = Figure(figsize=(6,11), dpi=70)
    a = figure.add_subplot(111)
    a.plot(x, y, color="green")
    a.set_title (ttl, fontsize=16)
    a.set_ylabel(yl, fontsize=14)
    a.set_xlabel(xl, fontsize=14)
    
    return figure

# here we define a class to plot the greeks
class Greeks():
    def __init__(self, S, K, T, t, q, r, s):
        self.S = S
        self.K = K
        self.t = T - t
        self.s = s
        self.r = r
        self.q = q
    
    # d1 of the Black and Scholes model for european options greeks closed formulas
    def d1(self, x):
        return (np.log(x / self.K) + (self.r - self.q + 0.5 * (self.s**2)) * self.t) / (self.s * np.sqrt(self.t))
    
    # d1 of the Black and Scholes model for european options greeks closed formulas    
    def d2(self, x):
        return (np.log(x / self.K) + (self.r - self.q - 0.5 * (self.s**2)) * self.t) / (self.s * np.sqrt(self.t))
    
    # the names of the functions are self explanatory
    def Delta(self, tp, cp):
        # if the option is EUROPEAN we have a closed formula
        if tp == "E" or (cp == "Call" and self.q == 0):
            x = np.append(np.linspace(self.S / 100, self.S * 2.5, 1000),np.array([self.S]))
            # call formula
            if cp == "Call":
                y = np.exp(-self.q * self.t) * norm.cdf(self.d1(x))
            # put formula
            elif cp == "Put":
                y = -np.exp(-self.q * self.t) * norm.cdf(-self.d1(x))
            return (plot(x[:-1], y[:-1], "Delta", "", ""), y[-1])
        # if the option is AMERICAN we use the central difference estimator
        elif tp == "A":
            x = np.linspace(self.S / 100, self.S * 2.5, 1000)
            h = x[1] - x[0]
            values = np.array([binomial_model(u, self.K, self.t, 0, self.q, self.r, self.s, 50, "American", cp) for u in x])
            delta = np.array([(values[i + 2] - values[i]) / (2 * h) for i in range(values.size - 2)])
            x = x[1 : len(x) - 1]
            
            delta_curr = binomial_model(self.S + h, self.K, self.t, 0, self.q, self.r, self.s, 50, "American", cp)
            delta_curr -= binomial_model(self.S - h, self.K, self.t, 0, self.q, self.r, self.s, 50, "American", cp)
            delta_curr /= 2 * h
            
            return (plot(x[:-1], delta[:-1], "Delta", "", ""), delta_curr)
            
    def Gamma(self, tp, cp):
        if tp == "E" or (cp == "Call" and self.q == 0):
            x = np.append(np.linspace(self.S / 100, self.S * 2.5, 1000),np.array([self.S]))
            # here we don't need two formulas since gamma is the same for calls and puts
            y = np.exp(-self.q * self.t) * norm.pdf(self.d1(x)) / (x * self.s * np.sqrt(self.t))
            return (plot(x[:-1], y[:-1], "Gamma", "", ""), y[-1])
        elif tp == "A":
            x = np.linspace(self.S / 100, self.S * 2.5, 1000)
            h = x[1] - x[0]
            values = np.array([binomial_model(u, self.K, self.t, 0, self.q, self.r, self.s, 50, "American", cp) for u in x])
            delta = np.array([(values[i + 2] - values[i]) / (2 * h) for i in range(values.size - 2)])
            gamma = np.array([(delta[i + 2] - delta[i]) / (2 * h) for i in range(delta.size - 2)])
            x = x[2 : len(x) - 2]
            
            delta_curr = binomial_model(self.S + h, self.K, self.t, 0, self.q, self.r, self.s, 50, "American", cp)
            delta_curr -= binomial_model(self.S - h, self.K, self.t, 0, self.q, self.r, self.s, 50, "American", cp)
            delta_curr /= 2 * h
            
            delta_curr_2 = binomial_model(self.S + (2 * h), self.K, self.t, 0, self.q, self.r, self.s, 50, "American", cp)
            delta_curr_2 -= binomial_model(self.S, self.K, self.t, 0, self.q, self.r, self.s, 50, "American", cp)
            delta_curr_2 /= 2 * h
            
            gamma_curr = (delta_curr_2 - delta_curr) / (2 * h)
            
            return (plot(x[:-1], gamma[:-1], "Gamma", "", ""), gamma_curr)
    
    def Rho(self, tp, cp):
        x = np.append(np.linspace(self.S / 100, self.S * 2.5, 1000),np.array([self.S]))
        if tp == "E" or (cp == "Call" and self.q == 0):
            if cp == "Call":    
                y = 0.01 * self.K * self.t * np.exp(-self.r * self.t) * norm.cdf(self.d2(x))
            elif cp == "Put":
                y = 0.01 * -self.K * self.t * np.exp(-self.r * self.t) * norm.cdf(-self.d2(x))
            return (plot(x[:-1], y[:-1], "Rho", "", ""), y[-1])
        elif tp == "A":
            # the plot will be Rho on the y axis and S on the x axis
            # therefore we calculate the derivative with respect to r
            # for each value of S (this is done also in the other greeks)
            h = 0.001
            rho = 0.01 * np.array([(binomial_model(u, self.K, self.t, 0, self.q, self.r + h, self.s, 50, "American", cp) - binomial_model(u, self.K, self.t, 0, self.q, self.r - h, self.s, 50, "American", cp)) / (2 * h) for u in x])
            return (plot(x[:-1], rho[:-1], "Rho", "", ""), rho[-1])
    
    def Vega(self, tp, cp):
        x = np.append(np.linspace(self.S / 100, self.S * 2.5, 1000),np.array([self.S]))
        if tp == "E" or (cp == "Call" and self.q == 0):
            y = 0.01 * x * np.exp(-self.q * self.t) * np.sqrt(self.t) * norm.pdf(self.d1(x))
            return (plot(x[:-1], y[:-1], "Vega", "", ""), y[-1])
        elif tp == "A":
            h = 0.001
            vega = 0.01 * np.array([(binomial_model(u, self.K, self.t, 0, self.q, self.r, self.s + h, 50, "American", cp) - binomial_model(u, self.K, self.t, 0, self.q, self.r, self.s - h, 50, "American", cp)) / (2 * h) for u in x])
            return (plot(x[:-1], vega[:-1], "Vega", "", ""), vega[-1])
    
    def Theta(self, tp, cp):
        x = np.append(np.linspace(self.S / 100, self.S * 2.5, 1000),np.array([self.S]))
        if tp == "E" or (cp == "Call" and self.q == 0):
            if cp == "Call":
                y = -np.exp(-self.q * self.t) * x * norm.pdf(self.d1(x)) * self.s / (2 * np.sqrt(self.t))
                y -= self.r * self.K * np.exp(-self.r * self.t) * norm.cdf(self.d2(x))
                y += self.q * x * np.exp(-self.q * self.t) * norm.cdf(self.d1(x))
                y /= 252
            elif cp == "Put":
                y = -np.exp(-self.q * self.t) * x * norm.pdf(-self.d1(x)) * self.s / (2 * np.sqrt(self.t))
                y += self.r * self.K * np.exp(-self.r * self.t) * norm.cdf(-self.d2(x))
                y -= self.q * x * np.exp(-self.q * self.t) * norm.cdf(-self.d1(x))
                y /= 252
            return (plot(x[:-1], y[:-1], "Theta", "", ""), y[-1])
        elif tp == "A":
            h = 0.001
            theta = np.array([(binomial_model(u, self.K, self.t - h, 0, self.q, self.r, self.s, 50, "American", cp) - binomial_model(u, self.K, self.t + h, 0, self.q, self.r, self.s, 50, "American", cp)) / (2 * h) for u in x]) / 252
            return (plot(x[:-1], theta[:-1], "Theta", "", ""), theta[-1])