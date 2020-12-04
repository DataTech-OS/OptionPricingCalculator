from scipy.integrate import quad
import numpy as np
from scipy.stats import norm

def black_scholes_model(S0, K, T, t, q, r, sigma, put_b):    
    if T == t and put_b == "Call":
        return max(S0 - K, 0)
    if T == t and put_b == "Put":
        return max(K - S0, 0)
    
    d1 = (np.log(S0 / K) + (r - q + ((sigma**2) / 2)) * (T - t)) / (sigma * np.sqrt(T - t)) 
    d2 = d1 - (sigma * (np.sqrt(T - t)))
	
    if put_b == "Call":
        return S0 * np.exp(-q * (T - t)) * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    if put_b == "Put":
        return K * np.exp(-r * (T - t)) * norm.cdf(-d2) -  S0 * np.exp(-q * (T - t)) * norm.cdf(-d1)
    
def binomial_model(S0, K, T, t, qd, r, sigma, nsteps, am_b, put_b):
    if T == t and put_b == "Call":
        return max(S0 - K,0)
    if T == t and put_b == "Put":
        return max(K - S0,0)
    
    nsteps = int(nsteps)
    S = np.zeros((nsteps, nsteps))
    X = np.zeros((nsteps, nsteps))
    
    S[0, 0] = S0
    
    deltat = (T - t) / nsteps
    u = np.exp(sigma * np.sqrt(deltat))
    d = 1 / u
    q = (np.exp((r - qd) * deltat) - d) / (u - d)
    
    for i in range(1, nsteps):
        for j in range(i + 1):
             if j == 0:
                S[0, i] = S[0, i - 1] * u
             else:
                S[j, i] = S[j - 1, i - 1] * d
            
    for j in range(nsteps):
        if put_b == "Call":
            X[j, nsteps - 1] = max(0, S[j, nsteps - 1] - K)
        elif put_b == "Put":
            X[j, nsteps - 1] = max(0, K - S[j, nsteps - 1])    
            
    for i in range(nsteps - 2, -1, -1):
        for j in range(i + 1):
            if am_b == "European":
                X[j, i] = np.exp(-r * deltat) * (q * X[j, i + 1] + (1 - q) * X[j + 1, i + 1])
            else:
                cv = np.exp(-r * deltat) * (q * X[j, i + 1] + (1 - q) * X[j + 1, i + 1])
                if put_b == "Call":
                    sv = max(0, S[j, i] - K)
                elif put_b == "Put":
                    sv = max(0, K - S[j, i])
                X[j, i] = max(cv, sv)

    return X[0, 0]

class Heston():
    def __init__(self, kappa, theta, vvol, rho, V0):
        self.kappa = kappa
        self.theta = theta
        self.vvol = vvol
        self.rho = rho
        self.V0 = V0 
        
        self.a = self.kappa * self.theta
        self.vol_risk_premium = 0
        self.b = self.kappa + self.vol_risk_premium
        
    def d(self, phi):
        return np.sqrt((self.kappa - self.rho * self.vvol * phi * 1j)**2 + self.vvol**2 * (phi * 1j + phi**2))
        
    def g(self, phi):
        return (self.kappa - self.rho * self.vvol * phi * 1j - self.d(phi)) / (self.kappa - self.rho * self.vvol * phi * 1j + self.d(phi))
        
    def C(self, phi):
        return (self.r - self.q) * phi * self.T * 1j + (self.a / (self.vvol**2)) * ((self.b - self.rho * self.vvol * phi * 1j - self.d(phi)) * self.T - 2 * np.log((1 - self.g(phi) * np.exp(-self.d(phi) * self.T)) / (1 - self.g(phi))))
        
    def D(self, phi):
        return ((self.kappa - self.rho * self.vvol * phi * 1j - self.d(phi)) / (self.vvol**2)) * (1 - np.exp(-self.d(phi) * self.T)) / (1 - self.g(phi) * np.exp(-self.d(phi) * self.T))    

    def char_f(self, phi):
        return np.exp(self.C(phi) +self.D(phi) * self.V0 + 1j * phi * np.log(self.S))
    
    def p1_f(self, x):
        return (np.exp(-1j * x * np.log(self.K)) * self.char_f(x - 1j)) / (1j * x * self.char_f(-1j))
    
    def p2_f(self, x):
        return (np.exp(-1j * x * np.log(self.K)) * self.char_f(x)) / (1j * x)  
    
    def price(self, S, K, T, r, q, tp):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        
        P1 = (1 / np.pi) * quad(lambda u: float(np.real(self.p1_f(u))),0,2104)[0]
        P2 = (1 / np.pi) * quad(lambda u: float(np.real(self.p2_f(u))),0,2104)[0]
    
        if tp == "Call":
            return S * np.exp(-q * T) * (0.5 + P1) - K * np.exp(-r * T) * (0.5 + P2)
        elif tp == "Put":
            return K * np.exp(-r * T) * (0.5 - P2) - S * np.exp(-q * T) * (0.5 - P1)
        
def heston_model(S, K, T, t, q, r, kappa, theta, vvol, rho, V0, tp):
    hm = Heston(kappa, theta, vvol, rho, V0)
    return hm.price(S, K, T - t, r, q, tp)

def Poisson(lambd, n, T):
    return np.exp(-lambd * T) * (lambd * T) ** n / np.math.factorial(n)

def jump_diffusion_model(S, K, T, t, q, r, sigma, lambd, a, b2, tp):
    c = 0
    price = 0
    n = 0
    m = np.exp(a + 0.5 * b2) - 1
    lambdap = lambd * (1 + m)
    T -= t
    
    while c < 0.999:
        rn = r - m * lambd + (n * np.log(1 + m) / T)
        sigman = np.sqrt(((sigma**2) * T + n * b2) / T)
        
        d1 = (np.log(S / K) + (rn - q + (sigman**2) / 2) * T) / (sigman * np.sqrt(T))
        d2 = d1 - sigman * np.sqrt(T)
        
        if tp == "Call":
            bs = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-rn * T) * norm.cdf(d2)
        elif tp == "Put":
            bs = K * np.exp(-rn * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        p = Poisson(lambdap, n, T)
        price = price + p * bs
        c = c + p
        n = n + 1
    
    return price