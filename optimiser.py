from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import minimize
from scipy.stats import norm
from numpy import log,sqrt,exp,inf,pi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
N,n = norm.cdf,norm.pdf
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')

class SSVI:
    def __init__(self):
        self.slice_errors = dict() #Mean error per slice

    def correlation(self,theta,a,b,c):  #skew term structure
        return a*exp(-b*theta)+c
    
    def power_law(self,theta,eta,gamma): 
        return eta/theta**gamma*(1+theta)**(1-gamma)
    
    def fly_constraint1(self,params,theta):
        a,b,c,eta,gamma = params
        rho = self.correlation(theta,a,b,c)
        phi = self.power_law(theta,eta,gamma)
        return (theta*phi*(1+abs(rho)))+4
    
    def eta_constraint1(self,params,theta):
        a,b,c,eta,gamma = params
        rho = self.correlation(theta,a,b,c)
        return eta+(4*max(theta)**(gamma-1))/(1+abs(rho))

    def fly_constraint2(self,params,theta): 
        a,b,c,eta,gamma = params
        rho = self.correlation(theta,a,b,c)
        phi = self.power_law(theta,eta,gamma)
        return (theta*phi**2*(1+abs(rho)))+(4+1e-10)  

    def eta_constraint3(self,params,theta):
        a,b,c,eta,gamma = params
        rho = self.correlation(theta,a,b,c)
        return eta*(1-abs(rho))+(2+1e-10)
    
    def eta_constraint2(self,params,theta):
        a,b,c,eta,gamma = params
        rho = self.correlation(theta,a,b,c)
        return eta+(2*max(theta)**(gamma-1/2))/(sqrt(1+abs(rho)))
    
    def rho_constraint(self,params,theta):
        a,b,c,eta,gamma = params
        rho = self.correlation(theta,a,b,c)
        return abs(rho)+1
    
    def gamma_constraint(self,params,theta):
        a,b,c,eta,gamma = params
        rho = self.correlation(theta,a,b,c)
        return gamma+((1+sqrt(1-rho**2))/rho**2)
    
    def abc_constraint(self,params):
        a,b,c,eta,gamma = params
        return abs((c-a)/(1-gamma*exp(gamma-2)))+1
    
    def ac_constraint(self,params):
        a,b,c,eta,gamma = params
        return abs(a+c)+1

    def surface(self,k,theta,a,b,c,eta,gamma):
        rho = self.correlation(theta,a,b,c)
        phi = self.power_law(theta,eta,gamma)
        return (theta/2)*(1+rho*phi*k+sqrt((k*phi+rho)**2+(1-rho**2)))
    
    def residual(self,params,k,theta,T,s):
        fit_vol,mid_vol = sqrt(self.surface(k,theta,*params)/(T/365)),s
        epsilon = [abs(mv-fv) for mv,fv in zip(mid_vol,fit_vol)]
        self.slice_errors = dict()
        for t in list(set(T)): #Total absolute errors in vols per slice / num of k
            idx = list(i for i,theta in enumerate(T) if theta==t)
            self.slice_errors[t] = (sum(np.array(epsilon)[idx])/len(idx))
        return sum(epsilon)**2 if np.isnan(sum(epsilon)) == False else 1e3
    
    #Fit paramterised vols to quoted vols
    def fit_vols(self,k,theta,T,s):
        guess = (-0.2,10,-0.2,0.5,0.25)
        bounds = ((-inf,inf),(0,inf),(-1,1),(0,1),(0,0.5))
        params = minimize(self.residual,guess,args=(k,theta,T,s),method="SLSQP",options={"maxiter":5000},bounds=bounds,
        constraints=({"type":"ineq","fun":self.eta_constraint1,"args":(theta,)},
                     {"type":"ineq","fun":self.eta_constraint2,"args":(theta,)},
                     {"type":"ineq","fun":self.eta_constraint3,"args":(theta,)},
                     {"type":"ineq","fun":self.fly_constraint1,"args":(theta,)},
                     {"type":"ineq","fun":self.fly_constraint2,"args":(theta,)},
                     {"type":"ineq","fun":self.rho_constraint,"args":(theta,)},
                     {"type":"ineq","fun":self.gamma_constraint,"args":(theta,)},
                     {"type":"ineq","fun":self.abc_constraint},
                     {"type":"ineq","fun":self.ac_constraint}))
        return params.x,self.slice_errors

    #Taking derivatives of total variances to logstrike
    def rnd(self,k,theta,a,b,c,eta,gamma):
        w = (theta/2)*(1+(a*exp(-b*theta)+c)*(eta/theta**gamma*(1+theta)**(1-gamma))*k+sqrt((k*(eta/theta**gamma*(1+theta)**(1-gamma))+(a*exp(-b*theta)+c))**2+(1-(a*exp(-b*theta)+c)**2)))
        wp = (theta/2)*((eta*(theta+1)**(1-gamma)*((eta*(theta+1)**(1-gamma)*k)/theta**gamma+a*exp(-b*theta)+c))/(theta**gamma*sqrt(((eta*(theta+1)**(1-gamma)*k)/theta**gamma+a*exp(-b*theta)+c)**2-(a*exp(-b*theta)+c)**2+1))+(eta*(theta+1)**(1-gamma)*(a*exp(-b*theta)+c))/theta**gamma)
        wpp = -(eta**2*theta**(1-2*gamma)*(theta+1)**(2-2*gamma)*exp(-2*b*theta)*((c**2-1)*exp(2*b*theta)+2*a*c*exp(b*theta)+a**2))/(2*(((eta*(theta+1)**(1-gamma)*k)/theta**gamma+a*exp(-b*theta)+c)**2-(a*exp(-b*theta)+c)**2+1)**(3/2))
        g = (1-0.5*(k*wp/w))**2-(0.25*wp**2)*(w**-1+0.25)+0.5*wpp
        d2 = (-k/sqrt(w)-0.5*sqrt(w))
        return (g/(sqrt(2*pi*w)))*exp(-0.5*d2**2)
    
    #Get intuitive params from raw ones
    def jumpwings(self,t,theta,a,b,c,eta,gamma):
        phi = self.power_law(theta,eta,gamma)
        rho = self.correlation(theta,a,b,c)
        atmvar = theta/t
        atmskew = 1/2*rho*sqrt(theta)*phi
        pslope = 1/2*sqrt(theta)*phi*(1-rho)
        cslope = 1/2*sqrt(theta)*phi*(1+rho)
        minvar = theta/t*(1-rho**2)
        return atmvar,atmskew,pslope,cslope,minvar
#---------------------------------------------------------------------------------------------
class BSM:
    def __init__(self,R,r,q,s):
        self.R,self.s,self.r,self.q = R,s,r,q

    def d1(self,x,y,t):
        s,r,q = self.s,self.r,self.q
        return (log(x/y)+(r-q+(s**2)/2)*t)/(s*sqrt(t))
    
    def d2(self,x,y,t):
        s,r,q = self.s,self.r,self.q
        return (log(x/y)+(r-q-(s**2)/2)*t)/(s*sqrt(t))

    def v(self,x,y,t):
        r,q,d1,d2 = self.r,self.q,self.d1,self.d2
        if self.R == "C":
            return (x*exp(-q*t)*N(d1(x,y,t)))-(y*exp(-r*t)*N(d2(x,y,t)))
        else:
            return (y*exp(-r*t)*N(-d2(x,y,t)))-(x*exp(-q*t)*N(-d1(x,y,t)))    
#--------------------------------------------------------------------------------------------- 
def density_moments(K,P,S,T):
    mean = sum([k*p for k,p in zip(K,P)])
    variance = sum([p*(k-mean)**2 for k,p in zip(K,P)])
    skewness = sum([p*((k-mean)/sqrt(variance))**3 for k,p in zip(K,P)])
    kurtosis = sum([p*((k-mean)/sqrt(variance))**4 for k,p in zip(K,P)])
    stddev = (sqrt(variance)/S)/sqrt(T)
    return mean,variance,skewness,kurtosis,stddev

#Total variances do not intersect = no calendar arb; density is non-negative = no fly arb
def raw2wings(a,b,m,V,T,F,S,params):
    K = np.linspace(a,b,m)
    jumpwings = pd.DataFrame(columns=["Tenor","AtmVar","AtmSkew","PWingSlope","CWingSlope","MinVar"])
    moments = pd.DataFrame(columns=["Tenor","Mean","Variance","Skewness","Kurtosis","StdDev"])
    for v,t,fwd in zip(V.unique(),T.unique(),F.unique()):
        k = log(K/fwd)
        f = spline(k,SSVI().rnd(k,v,*params),k=2)
        k = np.linspace(k[0],k[-1],m)
        P = [(k[1]-k[0])*p for p in f(k)]
        K = fwd*exp(k)
        moments.loc[len(moments)] = [t,*density_moments(K,P,S,t/365)]
        jumpwings.loc[len(jumpwings)] = [t,*SSVI().jumpwings(t/365,v,*params)]
    return jumpwings,moments

#---------------------------------------------------------------------------------------------     
def parametrise(df,S,F,k,theta,T,s,a,b,m):
    cols = ["Tenor","Time","SurfaceParams","SurfaceErrors","AtmVar","AtmSkew","PWingSlope","CWingSlope","MinVar","Mean","Variance","Skewness","Kurtosis","StdDev"]
    chi = pd.DataFrame(columns=cols)
    for t in df["Time"].unique():
        dfx = df[df["Time"]==t]
        raw_params,errors = SSVI().fit_vols(dfx[k],dfx[theta],dfx[T],dfx[s])
        moments,jumpwings = raw2wings(a,b,m,dfx[theta],dfx[T],dfx[F],dfx[S].iloc[0],raw_params)
        dfc = pd.DataFrame({"Tenor":list(errors.keys()),"Time":[t]*len(errors),
                            "SurfaceParams":[raw_params]*len(errors),"SurfaceErrors":list(errors.values())})
        dfc = dfc.set_index("Tenor").join(jumpwings.set_index("Tenor")).join(moments.set_index("Tenor")).reset_index()
        chi = chi.append(dfc)
    return df.set_index(["Time","Tenor"]).join(chi.set_index(["Time","Tenor"])).reset_index()
        
        
        
        
        
        
        