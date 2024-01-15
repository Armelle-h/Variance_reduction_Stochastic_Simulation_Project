import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as L #using the numpy module to generate the legendre polynomials
from scipy.stats import beta

#functions for the Least square Monte Carlo estimator
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

#functions for least square Monte Carlo estimator with importance sampling

def norm_legendre(j): #returns normalized legendre defined on [0,1]
    '''
    Returns the normalized jth Legendre polynomial defined on [0,1]
    
    input:
        j:int, the degree of the Legendre polynomial wanted
    output:
        the normalized jth Legendre polynomial on [0,1]
    '''
    return np.sqrt(2*j+1)*L.Legendre(degree(j), domain=[0,1])

def pdf_1_w(x, n):
    
    '''
    Computes the density function of the 1/w density evaluated at x defined for the sum of the first n+1 Legendre polynomials
    
    input:
        x: float in (0,1) the element at which we want to evaluate 1/w
        n: int where we consider the sum of the n first Legendre polynomials square in the def of 1/w
    output:
        the density function of the 1/w density evaluated at x defined for the sum of the first n+1 Legendre polynomials
    '''
    
    total=0
    for j in range(0,n+1):
        total+=norm_legendre(j)(x)**2
    return total/(n+1)

def l_2(x,i):
    '''
    evaluates the square of the ith normalized legendre polynomial at the value x
    
    input:
        x: float in (0,1) the element at which we want to evaluate the polynomial
        i: int, the degree of the wanted normalized Legendre polynomial
    output:
        the square of the ith normalized legendre polynomial evaluated at the value x
    '''
    return (2*i+1)*(L.Legendre(degree(i), domain=[0,1])(x) )**2

def gen_from_l_2(i, seed):
    '''
    Generates one random variables according to the ith legendre square distribution using the acceptance rejection method
    
    input:
        i: int, the degree of the wanted normalized Legendre polynomial
        seed: int, the seed for reproducibility
    output:
        (Y,V): a pair of floats where Y is generated as the squared ith Legendre polnomial
    '''
    np.random.seed(seed)
    
    up_bound=4*np.exp(1) 
    
    #rejection sampling 
    
    Y_generated=-999
    V_generated=-999 #initial values, haven't been updated yet
    
    a=1/2
    
    while Y_generated==-999: #while there was no update
        Y=np.random.beta(a, a)
        V=np.random.uniform(low=0.0, high=1)
        if V<l_2(Y,i)/(up_bound*beta.pdf(Y, a,a)):
            Y_generated=Y
            V_generated=V
    return Y, V #Y is the value of interest to us, V is for visualisation purposes

def gen_from_w(N, M, seed): #N the number of Legendre polynomials considered, M the number of wanted samples 
    '''
    Generates M random variables according to the density 1/w using the acceptance rejection method and the composition method
    
    input:
        N: int, the number of Legendre polynomials considered in the definition of 1/w
        M: int, the number of generated random variables
        seed: int, the seed for reproducibility
    output:
        a numpy array of M i.i.d random variables generated as 1/w
    '''    
    np.random.seed(seed)
    
    Y_generated=[]
    
    for m in range(M): #we generate M random variables
        index=np.random.randint(0, high=N+1)
        Y, _=gen_from_l_2(index, m)
        Y_generated.append(Y) #using m as the seed to have a seed varying and for reproducibility
        
    return np.array(Y_generated)

def w(x, N):
    '''
    the function w the inverse of the density 1/w
    '''
    return 1/pdf_1_w(x, N)

def coefficient_importance(M, N, seed):
    '''
     Computes the optimal solution of the least square approximation problem of f by the sum of the first N+1 Legendre polynomials using importance sampling
     
    input:
        M: int, the number of samples used by the least square Monte Carlo estimator
        N: int, where the function f is approximated by N+1 Legendre polynomials 
        seed: int the seed number for reproducibility
    output:
         c_coeff : array of size N+1, the optimal solution of the least square approximation problem using importance sampling of f by the sum of the first N+1 Legendre polynomials 
         cond_V_tilde: float, the condition number of V the coefficient matrix of the least square problem using importance sampling
    '''
    np.random.seed(seed) #for reproducibility
    X=np.random.uniform(size=M)
    V, _=V_mat(M, N, X)
    f_eval=np.array([f(x) for x in X])
    
    X_tilde= gen_from_w(N, M, seed)
    W_tilde=np.diag([np.sqrt(w(x_tilde, N)) for x_tilde in X_tilde])
    
    V_tilde=W_tilde@V
    cond_V_tilde=np.linalg.cond(V_tilde)
    
    f_eval_tilde=W_tilde@f_eval
    
    c_coeff,_,_,_=np.linalg.lstsq(V_tilde,f_eval_tilde, rcond=10**(-5))
    return c_coeff, cond_V_tilde

def IMCLS_importance(M, N, seed_1, seed_2):
    '''
    Computes the estimation error of the least square Monte Carlo and the least square Monte Carlo prime of the integral of the function f over the interval [0,1] using importance sampling
        
    input:
        M: int, the number of samples used by the least square Monte Carlo estimator
        N: int, where the function f is approximated by N+1 Legendre polynomials 
        seed_1: int, the seed number for reproducibility of the random variables used in the Monte Carlo least squares
        seed_2: int, the seed number for reproducibility of the random variables used in the computation of the optimal coefficient of the least square problem using importance sampling
    output:
         : float, the estimation error of the least square Monte Carlo estimator using importance sampling
         : float, the estimation error of the least square Monte Carlo estimator prime using importance sampling
         cond_V: float, the condition number of the coefficient matrix in the least square problem using importance sampling
    '''
    np.random.seed(seed_1)
    X=np.random.uniform(size=M)
    c_coeff, cond_V=coefficient_importance(M, N, seed_2)
    l_array=lambda x,n: np.array([np.sqrt(2*j+1)*L.Legendre(degree(j), domain=[0,1])(x) for j in range(0,n+1)])
    f_array=np.array([f(x)-c_coeff@l_array(x, N) for x in X])
    
    IMCLS_value=np.mean(f_array)+c_coeff[0]
    IMCLS_prime_val=c_coeff[0]
    true_val=np.arctan(5)/5
    
    return np.abs(IMCLS_value-true_val), np.abs(true_val-IMCLS_prime_val), cond_V