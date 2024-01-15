import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as L

def f(x):
    return 1/(25*x**2+1)

def Monte_Carlo(M, f, seed):
    '''
    Computes the estimation error of the Monte Carlo estimator of the integral of f over [0,1]
    
    input:
        M: int, the number of samples used by the Monte Carlo estimator
        f: a function of which we want to estimate its integral over the interval [0,1]
        seed: int the seed number for reproducibility
    output:
       an int, the estimation error of the Monte Carlo estimator of the integral of f over [0,1]
    '''
    np.random.seed(seed) #for reproducibility
    true_val=np.arctan(5)/5 #known true value of the integral
    
    X=np.random.uniform(size=M)
    
    f_array=np.array([f(x) for x in X])
    est_value=np.mean(f_array) #value of Monte Carlo estimation
    
    return np.abs(est_value-true_val)

def degree(n):
     '''
    Outputs a list with 0 and final value 1 used to call the nth Legendre polynomial in the module numpy.polynomial.legendre
    
    input:
        n: int
    output:
        a list of length n+1 of the shape [0,...,0,1] with n times the value 0 anf finishing by 1
    '''
    
    if n!=0:
        list_=[0]*(n+1)
        list_[-1]=1
        return list_
    if n==0:
        return [1,0] #small fix, using [1,0] to call the 0-Legendre polynomial instead of [1]
    
def V_mat(M, N, X):
     '''
     Computes the coefficient matrix associated with the least square approximation problem of f by the sum of the first N+1 Legendre polynomials evaluated at points in X
     
    input:
        M: int, the number of samples used by the least square Monte Carlo estimator
        N: int, where the function f is approximated by N+1 Legendre polynomials 
        X: array of size M where the function f is approximated by the Legendre polynomials evaluated at points in X
    output:
        V  : array of size M x (N+1) the Vandermonde matrix associated with the least square approximation problem of f by the sum of the first N+1 Legendre polynomials 
        conv_V: float, the condition number of V
    '''
        
    V=np.ones((M,N+1))
    for i in range(1,N+1):
        l_i=np.sqrt(2*i+1)*L.Legendre(degree(i), domain=[0,1]) #normalization coefficient to ensure orthonormality of
        #Legendre polynomials 
        V[:,i]=[l_i(x) for x in X]
    return V, np.linalg.cond(V)

def coefficient(M, N, seed):
     '''
     Computes the optimal solution of the least square approximation problem of f by the sum of the first N+1 Legendre polynomials 
     
    input:
        M: int, the number of samples used by the least square Monte Carlo estimator
        N: int, where the function f is approximated by N+1 Legendre polynomials 
        seed: int the seed number for reproducibility
    output:
         c_coeff : array of size N+1, the optimal solution of the least square approximation problem of f by the sum of the first N+1 Legendre polynomials 
         cond_V: float, the condition number of V the coefficient matrix of the least square problem
    '''
    np.random.seed(seed) #for reproducibility
    X=np.random.uniform(size=M)
    V, cond_V=V_mat(M, N, X) #defining the coefficient matrix of the least square problem
    f_eval=np.array([f(x) for x in X]) 
    c_coeff,_,_,_=np.linalg.lstsq(V,f_eval, rcond=10**(-5)) #solving the least square problem
    return c_coeff, cond_V

def IMCLS(M, N, seed_1, seed_2):
        '''
    Computes the estimation error of the least square Monte Carlo and the least square Monte Carlo prime of the integral of the function f over the interval [0,1]    
        
    input:
        M: int, the number of samples used by the least square Monte Carlo estimator
        N: int, where the function f is approximated by N+1 Legendre polynomials 
        seed_1: int, the seed number for reproducibility of the random variables used in the Monte Carlo least squares
        seed_2: int, the seed number for reproducibility of the random variables used in the computation of the optimal coefficient of the least square problem
    output:
         : float, the estimation error of the least square Monte Carlo estimator
         : float, the estimation error of the least square Monte Carlo estimator prime
         cond_V: float, the condition number of the coefficient matrix in the least square problem
    '''
    
    np.random.seed(seed_1)
    X=np.random.uniform(size=M)
    c_coeff, cond_V=coefficient(M, N, seed_2) #defining the coefficients used in the estimator
    l_array=lambda x,n: np.array([np.sqrt(2*j+1)*L.Legendre(degree(j), domain=[0,1])(x) for j in range(0,n+1)])
    f_array=np.array([f(x)-c_coeff@l_array(x, N) for x in X])
    
    IMCLS_value=np.mean(f_array)+c_coeff[0] #value of the MCLS estimator
    IMCLS_prime_val=c_coeff[0] #value of the MCLS' estimator
    true_val=np.arctan(5)/5 #insider knowledge, true value of the integral of f over [0,1]
    
    return np.abs(IMCLS_value-true_val), np.abs(true_val-IMCLS_prime_val), cond_V
