import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as L

def f(x):
    return 1/(25*x**2+1)

def Markov(M, f, seed):
    np.random.seed(seed) #for reproducibility
    true_val=np.arctan(5)/5
    
    X=np.random.uniform(size=M)
    
    f_array=np.array([f(x) for x in X])
    est_value=np.mean(f_array)
    
    return np.abs(est_value-true_val)

def degree(n):
    if n!=0:
        list_=[0]*(n+1)
        list_[-1]=1
        return list_
    if n==0:
        return [1,0] #small fix, [1,0] is equivalent to [1] but they didn't like [1]
    
def V_mat(M, N, X):
    V=np.ones((M,N+1))
    for i in range(1,N+1):
        l_i=np.sqrt(2*i+1)*L.Legendre(degree(i), domain=[0,1]) #normalization coefficient to ensure orthonormality of
        #Legendre polynomials 
        V[:,i]=[l_i(x) for x in X]
    return V, np.linalg.cond(V)

def coefficient(M, N, seed):
    np.random.seed(seed) #for reproducibility
    X=np.random.uniform(size=M)
    V, cond_V=V_mat(M, N, X)
    f_eval=np.array([f(x) for x in X])
    c_coeff,_,_,_=np.linalg.lstsq(V,f_eval, rcond=10**(-5))
    return c_coeff, cond_V

def IMCLS(M, N, seed_1, seed_2):
    
    np.random.seed(seed_1)
    X=np.random.uniform(size=M)
    c_coeff, cond_V=coefficient(M, N, seed_2)
    l_array=lambda x,n: np.array([np.sqrt(2*j+1)*L.Legendre(degree(j), domain=[0,1])(x) for j in range(0,n+1)])
    f_array=np.array([f(x)-c_coeff@l_array(x, N) for x in X])
    
    IMCLS_value=np.mean(f_array)+c_coeff[0]
    IMCLS_prime_val=c_coeff[0]
    true_val=np.arctan(5)/5
    
    return np.abs(IMCLS_value-true_val), np.abs(true_val-IMCLS_prime_val), cond_V
