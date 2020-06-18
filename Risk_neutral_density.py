from scipy.stats import norminvgauss
from scipy.optimize import curve_fit,minimize
from scipy.integrate import quad,simps
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from scipy.stats import norm,lognorm
from numpy import log,sqrt,exp,inf,pi
import scipy.stats as st
from scipy import signal
import pandas as pd
import scipy as sp
import numpy as np
from scipy.stats import invgauss,norminvgauss
cdf,pdf = norm.cdf,norm.pdf
pd.options.mode.chained_assignment = None
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')

def raw_svi(x,a,b,rho,m,sigma):
    return a+b*(rho*(x-m)+sqrt((x-m)**2+sigma**2))

def get_value(R,S,X,T,r,q,s): 
    T = T/365
    d1 = (log(S/X)+(r-q+(s**2)/2)*T)/(s*sqrt(T))
    d2 = d1-(s*sqrt(T))
    if R == "Call":
        return (S*exp(-q*T)*cdf(d1))-(X*exp(-r*T)*cdf(d2))
    else:
        return (X*exp(-r*T)*cdf(-d2))-(S*exp(-q*T)*cdf(-d1))

def mom_distance(K,Q,nQ,moment):
    moment_dict = {"M":"mean","V":"variance","S":"skewness","K":"kurtosis"}
    moment = moment_dict[moment]
    mean,variance,skewness,kurtosis = get_moments(K,nQ)
    nmean,nvariance,nskewness,nkurtosis = get_moments(K,Q)    
    return abs(vars()[moment]-vars()["n"+moment])

def mom_residual(params,K,T,Q,moment,target,restrictive): 
    nQ = rnd(K,T,S,r,q,*params)
    mean,variance,skewness,kurtosis = get_moments(K,nQ)
    hellinger_distance = sqrt(sum((sqrt(Q)-sqrt(nQ))**2))/sqrt(2)
    if restrictive == True: #Isolating scaling to selected moment
        if moment == "kurtosis":
            moment_distance = mom_distance(K,Q,nQ,"S")
        elif moment == "skewness":
            moment_distance = mom_distance(K,Q,nQ,"K")
        epsilon = abs(vars()[moment]-target)+hellinger_distance+moment_distance
    else: #Scaling can effect other moments
        epsilon = abs(vars()[moment]-target)+hellinger_distance
    return epsilon
    
def mom_fit(K,T,S,r,q,moment,factor,restrictive:bool,old_params):
    if moment not in ["S","K"]: print("Select either S or K"); return
    moment_dict = {"S":"skewness","K":"kurtosis"}
    moment = moment_dict[moment]
    k = get_logstrike(K,T,S,r,q)
    w = raw_svi(k,*old_params)
    Q = rnd(K,T,S,r,q,*old_params)
    mean,variance,skewness,kurtosis = get_moments(K,Q)
    target = vars()[moment]*factor
    print(f"Scaling {moment} by factor of {factor} with restrictive method set to {restrictive}")
    print(f"Old: Mean={mean:.3f} Variance={variance:.3f} Skewness={skewness:.3f} Kurtosis={kurtosis:.3f}")
    #Bounds
    a_ = (0,max(w)) #vertical translation
    b_ = (0,10) #wings slope
    rho_ = (-1,1) #counter-clockwise rotation
    m_ = (2*min(k),2*max(k)) #horizontal translation
    sigma_ = (0.1,10) #smile curvature
    #Imposing constraints (0<x<4 and 1<x) so that RND's P(x)>0 making smile free of fly arb
    def left_wing_first_coefficient_lhs(x): return ((x[1]**2)*(x[2]+1)**2)-0
    def left_wing_first_coefficient_rhs(x): return ((x[1]**2)*(x[2]+1)**2)+4
    def right_wing_first_coefficient_lhs(x): return ((x[1]**2)*(x[2]-1)**2)-0
    def right_wing_first_coefficient_rhs(x): return ((x[1]**2)*(x[2]-1)**2)+4
    def left_wing_discriminant(x): return (x[0]-x[3]*x[1]*(x[2]-1))*(4-x[0]+x[3]*x[1]*(x[2]-1))-(x[1]**2*(x[2]-1)**2)-0
    def right_wing_discriminant(x): return (x[0]-x[3]*x[1]*(x[2]+1))*(4-x[0]+x[3]*x[1]*(x[2]+1))-(x[1]**2*(x[2]+1)**2)-0
    def slope1(x,T): return (x[1])+(4/((T/365)*(1+abs(x[2]))))
    def slope2(x): return (x[1])-0
    new_params = minimize(mom_residual,(old_params),method="SLSQP",args=(K,T,Q,moment,target,restrictive),
                          options={"maxiter":2000},bounds=(a_,b_,rho_,m_,sigma_),
                          constraints=({"type":"ineq","fun":left_wing_first_coefficient_lhs},
                                       {"type":"ineq","fun":left_wing_first_coefficient_rhs},
                                       {"type":"ineq","fun":right_wing_first_coefficient_lhs},
                                       {"type":"ineq","fun":right_wing_first_coefficient_rhs},
                                       {"type":"ineq","fun":left_wing_discriminant},
                                       {"type":"ineq","fun":right_wing_discriminant},
                                       {"type":"ineq","fun":slope1,"args":[T,]},
                                       {"type":"ineq","fun":slope2})).x
    mean,variance,skewness,kurtosis = get_moments(K,rnd(K,T,S,r,q,*new_params))    
    print(f"New: Mean={mean:.3f} Variance={variance:.3f} Skewness={skewness:.3f} Kurtosis={kurtosis:.3f}")
    mom_graph(K,T,S,r,q,old_params,new_params)
    return new_params

def mom_graph(K,T,S,r,q,old_params,new_params):
    returns = np.array([(k/S)-1 for k in K])[1:-1]
    new_density = rnd(K,T,S,r,q,*new_params)
    old_density = rnd(K,T,S,r,q,*old_params)
    rwd_density = rwd(K,T,old_density,S,r,q)
    print(sum(rwd_density))
    plt.plot(returns,new_density,c="g")
    plt.plot(returns,old_density,c="r")
    plt.plot(returns,rwd_density,c="b")
    plt.plot([0,0],[0,max(max(new_density),max(old_density))],c="k",linewidth=1)
    #plt.ylim(0,max(max(new_density),max(old_density)))
    plt.show()
    
    plt.scatter(sorted(old_density),sorted(new_density),s=1,c="k",alpha=0.2)
    plt.plot([0,max(old_density)],[0,max(new_density)],c="k",linewidth=1)
    plt.xlim(0,max(old_density))
    plt.ylim(0,max(new_density))
    plt.show()
    
def get_moments(X,P):
    mean = sum([x*p for x,p in zip(X,P)])
    variance = sum([p*(x-mean)**2 for x,p in zip(X,P)])
    skewness = sum([p*((x-mean)/sqrt(variance))**3 for x,p in zip(X,P)])
    kurtosis = sum([p*((x-mean)/sqrt(variance))**4 for x,p in zip(X,P)])-3
    #implied_vol = (sqrt(variance)/S)/sqrt(T/365)
    return mean,variance,skewness,kurtosis

def rwd(K,T,Q,S,r,q):
    return [exp((q-r)*T/365)*(k/S)*p for k,p in zip(K,Q)]

def NIG(x,alpha,beta,mu,delta):
    gamma = sqrt(alpha**2-beta**2)
    i = alpha*sqrt(delta**2+(x-mu)**2)
    K1 = sp.special.kn(1,i)
    exponent = exp(delta*gamma+beta*(x-mu))
    numerator = alpha*delta*K1
    denominator = pi*sqrt(delta**2+(x-mu)**2)
    p = (numerator/denominator)*exponent
    return p

#Breeden&Litzenberger density: taking second derivative of call price to strike
def rnd(K,T,S,r,q,a,b,rho,m,sigma):
    dK = (K[1]-K[0])
    k = get_logstrike(K,T,S,r,q)
    C = get_value("Call",S,K,T,r,q,sqrt(raw_svi(k,a,b,rho,m,sigma)))
    Q = [exp(r*(T/365))*(c2-(2*c1)+c0)/dK for c2,c1,c0 in zip(C[2:],C[1:-1],C[:-2])]
    #K,P = map(list,zip(*[[k,p] for k,p in zip(K[1:-1],P) if p>1e-6]))
    return Q

def get_strike_vector(dK,maxK): return np.array([i*dK for i in range(1,int(maxK/dK)+3)])

def get_logstrike(K,T,S,r,q): return log(K/(S*exp((-r-q)*(T/365))))

#Load up SVI params   
chi = pd.read_csv("C:/Users/aigar/Desktop/Scripts/params.csv").dropna().iloc[1] #XXXXXXXXXXX
T,S,r,q = chi["t"],chi["Spot"],chi["Rate"],chi["q"]
params = chi[3:8]

K = get_strike_vector(0.1,60)
w = raw_svi(get_logstrike(K,T,S,r,q),*params)
spreads = [0+(i) for i in range(602)]

Q = rnd(K,T,S,r,q,*params)
returns = np.array([(k/S)-1 for k in K])[1:-1]

#Get close prices
df = pd.read_csv("C:/Users/aigar/Desktop/Scripts/Book2.csv")
r = sorted(set(np.array(df["Close"])))
avg = sum(r)/len(r)

params = norminvgauss.fit(r) #MLE - fit NIG distribution to close prices
P = NIG(K,*params)
P = norminvgauss.pdf(K,*params)
P = [i*0.1 for i in P][1:-1]

plt.plot(returns,P,c="g") #Realised density - P measure
plt.plot(returns,Q,c="r") #Implied density - Q measure
plt.show()

#Convolve P & Q measures to get implied pricing kernel
filtered = signal.convolve(P,Q,mode="same",method="fft")/sum(Q)
plt.plot(returns,filtered)
plt.show()
