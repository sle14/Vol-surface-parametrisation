from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import curve_fit,bisect,brentq,least_squares
from scipy.interpolate import splrep, splev
from numpy import log,sqrt,exp,inf,nan,pi
from numpy.ctypeslib import ndpointer
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import stats
from time import sleep
import concurrent.futures
import numpy as np
import ctypes
import utils
import os

cout = utils.log(__file__,__name__,disp=False)
N,n = norm.cdf,norm.pdf
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')
base = 253

#------------------------------------------------------------------------------------------------   
curr_dir = os.path.dirname(os.path.realpath(__file__))
lib = ctypes.cdll.LoadLibrary(f"{curr_dir}/KimIntegral.dll")
class KimIntegral:
    def __init__(self,R,S,K,T,s,r,q,H0,H1,Kh,M=12):
        R = 1 if R=="C" or R=="Call" or R==1 else 0
        lib.KI.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,
                           ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,
                           ctypes.c_double,ctypes.c_double,ctypes.c_double]
        lib.KI.restype = ctypes.c_void_p
        lib.fit_boundary.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_bool]
        lib.fit_boundary.restype = ndpointer(dtype=ctypes.c_double,shape=(M+1,))
        lib.get_value.argtypes = [ctypes.c_void_p,ndpointer(dtype=ctypes.c_double,shape=(M+1,))]
        lib.get_value.restype = ndpointer(dtype=ctypes.c_double,shape=(2,))
        self.obj = lib.KI(R,M,S,K,T,s,r,q,H0,H1,Kh)

    def fit_boundary(self,n,spread):
        return lib.fit_boundary(self.obj,n,spread)
    
    def get_value(self,B):
        ame = lib.get_value(self.obj,B)
        return ame[0],round(ame[1],6)

class ThreadPool:
    def __init__(self,R,S,K,T,s,r,q,H0,H1,Kh):
        R = 1 if R=="C" or R=="Call" or R==1 else 0
        self.R,self.S,self.K,self.T = R,S,K,T
        self.s,self.r,self.q = s,r,q
        self.H0,self.H1,self.Kh = H0,H1,Kh
        self.EUR = np.ones(len(K)) * -100
        self.EEP = np.ones(len(K)) * -100
        if len(K) % 4 == 0:
            self.idx = int(len(K)/4)
            self.workers = 4
        elif len(K) % 3 == 0:
            self.idx = int(len(K)/3)
            self.workers = 3
        elif len(K) % 2 == 0:
            self.idx = int(len(K)/2)
            self.workers = 2
        else:
            if (len(K)-1) % 4 == 0:
                self.idx = int((len(K)-1)/4)
                self.workers = 4
            elif (len(K)-1) % 3 == 0:
                self.idx = int((len(K)-1)/3)
                self.workers = 3
            elif (len(K)-1) % 2 == 0:
                self.idx = int((len(K)-1)/2)
                self.workers = 2            
            else:
                print(f"Kaput for {len(K)} array")
        #print(f"Workers: {self.workers}, Idx: {self.idx}, Array: {len(K)}")
        # self.workers = None
        
    def func(self,worker):
        sleep(0.4*worker)
        for i in range(self.idx):
            idx = int(worker*self.idx)+i
            KI = KimIntegral(self.R,self.S[idx],self.K[idx],self.T[idx],self.s[idx],self.r[idx],self.q[idx],self.H0[idx],self.H1[idx],self.Kh[idx])
            self.EUR[idx],self.EEP[idx] =  KI.get_value(KI.fit_boundary(200,True))
            
    def remainder(self):
        eur_idx = np.where(self.EUR<-99)
        eep_idx = np.where(self.EEP<-99)
        for i in np.unique(np.union1d(eur_idx,eep_idx)):
            KI = KimIntegral(self.R,self.S[i],self.K[i],self.T[i],self.s[i],self.r[i],self.q[i],self.H0[i],self.H1[i],self.Kh[i])
            eur,eep = KI.get_value(KI.fit_boundary(200,True))
            self.EUR[i] = round(eur,3)
            self.EEP[i] = round(eep,3)
            
    def run(self):
        #start = datetime.now()
        if self.workers is not None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                executor.map(self.func,range(self.workers))
        while True:
            if len(self.EUR[self.EUR<-99]) > 0 or len(self.EEP[self.EEP<-99]) > 0:#
                self.remainder()
            else:
                break
        #print(f"exec: {datetime.now()-start}s")
        return self.EUR,self.EEP
        
    
#------------------------------------------------------------------------------------------------       
def trapz(f,a,b,n):
    x = np.linspace(a,b,n)
    y = np.nan_to_num(f(x))
    h = (b-a)/(n-1)
    return (h/2)*(y[1:]+y[:-1]).sum()

class BlackScholes:
    def __init__(self,R,r,q,s=None):
        self.R,self.s,self.r,self.q = R,s,r,q

    def d1(self,x,y,t):
        s,r,q = self.s,self.r,self.q
        return (log(x/y)+(r-q+(s**2)/2)*t)/(s*sqrt(t))
    
    def d2(self,x,y,t):
        s,r,q = self.s,self.r,self.q
        return (log(x/y)+(r-q-(s**2)/2)*t)/(s*sqrt(t))

    def vd(self,x,y,t,D,td): #Eur value with discrete div 
        s,r = self.s,self.r
        def f(z):
            xd = x*exp((r-(s**2)/2)*td + z*s*sqrt(td)) - D
            return self.v(xd,y,t-td)*n(z)
        a,b,m = (log(D/x)-(r-(s**2)/2)*td)/(s*sqrt(td)),x*5,5000
        return trapz(f,a,b,m)*exp(-r*td)
    
    def v(self,x,y,t):
        r,q,d1,d2 = self.r,self.q,self.d1,self.d2
        if self.R == "C":
            return (x*exp(-q*t)*N(d1(x,y,t)))-(y*exp(-r*t)*N(d2(x,y,t)))
        else:
            return (y*exp(-r*t)*N(-d2(x,y,t)))-(x*exp(-q*t)*N(-d1(x,y,t)))
        
    def delta(self,x,y,t):
        q,d1 = self.q,self.d1
        if self.R == "C":
            return exp(-q*t)*N(d1(x,y,t))
        else:
            return -exp(-q*t)*N(-d1(x,y,t))
    
    def gamma(self,x,y,t):
        q,s,d1 = self.q,self.s,self.d1
        return (exp(-q*t)*n(d1(x,y,t)))/(x*s*sqrt(t))
    
    def vega(self,x,y,t):
        q,d1 = self.q,self.d1
        return (x*exp(-q*t)*n(d1(x,y,t))*sqrt(t))/100
    
    def theta(self,x,y,t):
        r,q,s,d1,d2 = self.r,self.q,self.s,self.d1,self.d2
        if self.R == "C":
            return ((-exp(-q*t)*((x*s*n(d1(x,y,t)))/(2*sqrt(t)))) - (r*y*exp(-r*t)*N(d2(x,y,t))))/base
        else:
            return ((-exp(-q*t)*((x*s*n(-d1(x,y,t)))/(2*sqrt(t)))) + (r*y*exp(-r*t)*N(-d2(x,y,t))))/base

    def rho(self,x,y,t):
        r,d2 = self.r,self.d2
        if self.R == "C" :
            return (y*t*exp(-r*t)*N(d2(x,y,t)))/100
        else:
            return (-y*t*exp(-r*t)*N(-d2(x,y,t)))/100

    def vomma(self,x,y,t):
        q,s,d1,d2 = self.q,self.s,self.d1,self.d2
        return (((x*exp(-q*t)*n(d1(x,y,t))*sqrt(t))/100)*d1(x,y,t)*d2(x,y,t))/s

    def vanna(self,x,y,t):
        q,s,d1,d2 = self.q,self.s,self.d1,self.d2
        return exp(-q*t)*sqrt(t)*n(d1(x,y,t))*(d2(x,y,t)/s)

    def speed(self,x,y,t):
        q,s,d1 = self.q,self.s,self.d1
        return -exp(-q*t)*(n(d1(x,y,t))/(x**2*s*sqrt(t)))*(1+(d1(x,y,t)/(s*sqrt(t))))

    def charm(self,x,y,t):
        r,q,s,d1,d2 = self.r,self.q,self.s,self.d1,self.d2
        if self.R == "C":
            return q*exp(-q*t)*N(d1(x,y,t))-exp(-q*t)*n(d1(x,y,t))*(2*t*(r-q)-d2(x,y,t)*s*sqrt(t))/(2*t*s*sqrt(t))
        else:
            return -q*exp(-q*t)*N(-d1(x,y,t))-exp(-q*t)*n(d1(x,y,t))*(2*t*(r-q)-d2(x,y,t)*s*sqrt(t))/(2*t*s*sqrt(t))
        
#---------------------------------------------------------------------------------------------
class Vol:
    def __init__(self,R,P,S,K,T,r,q,H0,H1,Kh):
        self.R,self.P,self.S,self.K = R,P,S,K
        self.T,self.r,self.q = T/base,r,q
        self.H0,self.H1,self.Kh = H0,H1,Kh    
        self.vols = np.zeros(self.P.shape[0])
        self.eeps = np.zeros(self.P.shape[0])

    def eur_res(self,s,i):
        if s < 0: return 1e6
        BSM = BlackScholes(self.R,self.r[i],self.q[i],s)
        epsilon = self.P[i] - BSM.v(self.S[i],self.K[i],self.T[i])
        return epsilon if np.absolute(epsilon) > 1e-3 else 0

    def ame_res(self,s,i):
        if s < 0: return 1e6
        KI = KimIntegral(self.R,self.S[i],self.K[i],self.T[i],
             s,self.r[i],self.q[i],self.H0[i],self.H1[i],self.Kh[i])
        eeb = KI.fit_boundary(200,True)
        eur,eep = KI.get_value(eeb)
        epsilon = self.P[i] - (eur + eep)
        if np.absolute(epsilon) > 1e-6:
            return epsilon
        else:
            self.eeps[i] = eep
            return 0
    
    def root(self,a,b):
        cout.info(f"Solving {self.R} vol roots")
        for i in range(self.P.shape[0]):
            try: 
                x0 = bisect(self.eur_res,a,b,args=(i,))
                if self.R == "P" and self.r[0] < 0: 
                    self.vols[i] = x0
                else:
                    self.vols[i] = brentq(self.ame_res,x0/1.5,x0*1.1,args=(i,))
            except:
                self.vols[i],self.eeps[i] = np.nan,np.nan
            # print(f"{round(self.T[i]*base)}, {self.R}, {self.K[i]}, {self.vols[i]}, {self.eeps[i]}")
            cout.info(f"{round(self.T[i]*base)}, {self.R}, {self.K[i]}, {self.vols[i]}, {self.eeps[i]}")
        # print(f"Done {self.R} vol roots")
        cout.info(f"Done {self.R} vol roots")
        return self.vols,self.eeps   
#---------------------------------------------------------------------------------------------    
class SSVI:
    def __init__(self,return_errors=False):
        self.return_errors = return_errors
        if return_errors == True: self.epsilon = np.zeros(100000)
    
    def correlation(self,theta,a,b,c):                     #Skew term structure
        return a*exp(-b*theta)+c
    
    def power_law(self,theta,eta,lambda_):                 #Convexity function
        return eta*theta**-lambda_
   
    def raw2jw(self,t,theta,phi,rho):
        sigma = sqrt(theta/t)                              #Atmf vol
        psi = (1/2*rho*sqrt(theta)*phi)/t                  #Atmf variance skew
        kappa = (1/2*theta*phi**2*(1-rho**2))/t            #Atmf variance kurt
        return sigma,psi,kappa

    def jw2raw(self,t,sigma,psi,kappa):                   
        theta = sigma**2*t                                 #Atmf total variance
        psi = psi*t                                        #Atmf total skew
        kappa = kappa*t                                    #Atmf total kurt
        c = kappa/(2*(sqrt((kappa/2)+psi**2)-psi))
        rho = psi/(c-psi)
        phi = (2*psi)/(sqrt(theta)*rho)
        return theta,phi,rho
    
    def raw(self,k,theta,phi,rho):
        return (theta/2)*(1+rho*phi*k+sqrt((k*phi+rho)**2+(1-rho**2)))
    
    def rnd(self,k,theta,phi,rho):
        dk = abs(k[1]-k[0])
        w = (theta/2)*(1+rho*phi*k+sqrt((k*phi+rho)**2+(1-rho**2)))
        wp = (theta*((phi*(phi*k+rho))/sqrt((phi*k+rho)**2-rho**2+1)+phi*rho))/2
        wpp = -(phi**2*(rho**2-1)*theta)/(2*((phi*k+rho)**2-rho**2+1)**(3/2))
        g = (1-0.5*(k*wp/w))**2-(0.25*wp**2)*(1/w+0.25)+0.5*wpp
        d2 = -k/sqrt(w) + sqrt(w)/2
        p = n(d2)*(g/sqrt(w))
        return np.nan_to_num(p)*dk 
   
    def calendar1(self,params,theta):
        a,b,c,eta,lambda_ = params
        rho = self.correlation(theta,a,b,c)
        phi = self.power_law(theta,eta,lambda_)
        dtheta = np.insert(np.diff(theta),0,0.,axis=0)
        return dtheta*theta*phi + (1/(rho**2))*(1+sqrt(1-rho**2))*phi
    
    def butterfly1(self,params,theta):
        a,b,c,eta,lambda_ = params
        rho = self.correlation(theta,a,b,c)
        phi = self.power_law(theta,eta,lambda_)
        return (theta*phi*(1+abs(rho))) + 4

    def butterfly2(self,params,theta): 
        a,b,c,eta,lambda_ = params
        rho = self.correlation(theta,a,b,c)
        phi = self.power_law(theta,eta,lambda_)
        return (theta*phi**2*(1+abs(rho))) + (4+1e-10)  

    def rho1(self,params,theta):
        a,b,c,eta,lambda_ = params
        rho = self.correlation(theta,a,b,c)
        return abs(rho) + 1

    def residual(self,params,k,theta,T,s,weights):
        a,b,c,eta,lambda_ = params
        phi = self.power_law(theta,eta,lambda_)
        rho = self.correlation(theta,a,b,c)
        fit_vol,mid_vol = sqrt(self.raw(k,theta,phi,rho)/(T/base)),s
        epsilon = [abs(mv-fv)*w for mv,fv,w in zip(mid_vol,fit_vol,weights)]
        if self.return_errors == True: 
            abs_error = sum([abs(mv-fv) for mv,fv in zip(mid_vol,fit_vol)])
            self.epsilon[np.where(self.epsilon==0)[0][0]] = abs_error
        return sum(epsilon)**2 if np.isnan(sum(epsilon)) == False else 1e6

    def constrains(self,theta):
        return ({"type":"ineq","fun":self.calendar1,"args":(theta,)},
                {"type":"ineq","fun":self.butterfly1,"args":(theta,)},
                {"type":"ineq","fun":self.butterfly2,"args":(theta,)},
                {"type":"ineq","fun":self.rho1,"args":(theta,)})

    def calibrate(self,k,theta,T,s,weights):
        cout.info("Calibrating vols")
        params = minimize(
                          self.residual,
                          x0 = (-0.2,10,-0.2,0.5,0.45),
                          args = (k,theta,T,s,weights),
                          method = "SLSQP",
                          options = {"maxiter":5000},
                          bounds = ((-inf,inf),(0,inf),(-1,1),(0,1),(0,0.5)),
                          constraints = self.constrains(theta)
                         )
        if self.return_errors == True:
            return (params.x[0],params.x[1],params.x[2],params.x[3],params.x[4],self.epsilon[np.where(self.epsilon!=0)])
        else:
            return (params.x[0],params.x[1],params.x[2],params.x[3],params.x[4])

    def rho(self,params,phi,rho,theta):
        a,b,c = params
        return ((a*exp(-b*theta)+c) - rho)
    
    def phi(self,params,phi,rho,theta):  
        eta,lambda_ = params
        return ((eta*theta**-lambda_) - phi)

    def recalibrate(self,theta,phi,rho):
        res1 = least_squares(self.rho, (-0.2,10,-0.2), 
                             bounds = ((-inf,0,-1),(inf,inf,1)), 
                             args = (phi,rho,theta),
                             xtol = 1e-12).x
        res2 = least_squares(self.phi, (0.5,0.45), 
                             bounds = ((0,0),(1,0.5)), 
                             args = (phi,rho,theta),
                             xtol = 1e-12).x
        params = list(res1)+list(res2)
        return params

#------------------------------------------------------------------------------------------------        
def dsc2yld(S,T,D,Td,r):
    return -log((exp(-Td*r)*(S*exp(Td*r)-D))/S)/T

def yld2dsc(S,T,Td,q,r):
    return S*(exp(q*T)-1)*exp(Td*r-T*q)
    
def impdivsborrows(M,S,K,T,C,P,r,D,Td):
    cout.info("Deriving borrows")
    bor = -1/(T/base)*log((C-P+K*exp(-r*(T/base))+D*exp(-r*(Td/base)))/S)
    yld = -1/(T/base)*log((C-P+K*exp(-r*(T/base)))/S)
    idx = np.where(M>1)[0]
    bor = np.take(bor,idx,axis=0)
    bor = stats.trim_mean(bor,0.01)
    yld = np.take(yld,idx,axis=0)
    yld = stats.trim_mean(yld,0.01)
    return np.nan_to_num(bor),np.nan_to_num(yld)

def halfspread(x,H0,H1,Kh,R):
    return np.where(
                    (R=="Call")|(R=="C")|(R==1),
                    H0+H1*np.maximum(Kh-x,0),
                    H0+H1*np.maximum(x-Kh,0)
                   )/2
        
def fit_spread(K,S,spread,R):
    cout.info("Fitting spreads")
    H = lambda k,h0,h1,kh: halfspread(k,h0,h1,kh,R)*2
    std = np.std(spread)
    if len(K) > 5 and std > 0.01:
        mu = np.mean(spread)
        idx = np.where(abs(spread-mu) < (2 * std))
        spread = spread[idx]
        K = K[idx]
    popt,pcov = curve_fit(
                          H,K,spread,
                          p0 = (min(spread),1e-6,S[0]),
                          bounds = ((0,0,S[0]/3),(30,1,S[0]*3)),
                         )
    return popt[0],popt[1],popt[2]

def interp(k,s):
    cout.info("Interpolating vols")
    s_exnan = s[~np.isnan(s)]
    k_exnan = k[~np.isnan(s)]
    weights = n(k_exnan,0,0.3)
    if len(k_exnan) >= 6:
        bspl = splrep(k_exnan,s_exnan,w=weights,s=1)
        s_exnan = splev(k_exnan,bspl)
        f = spline(k_exnan,s_exnan,k=1)
        return f(k)
    elif 6 > len(k_exnan) >= 2:
        f = spline(k_exnan,s_exnan,k=1)
        return f(k)
    else:
        return s

def norm_weights(k,loc=0,scale=0):
    if scale == 0: scale = 0.1 #we want to fit on vol ex liquidity premium
    w = n(k,loc,scale)
    return w/max(w)
    
def forward(T,S,r,q,Td,D):
    cout.info("Deriving forwards")
    return (S-D*exp(-r*(Td/base)))*exp((r-q)*(T/base))

def totatmfvar(T,k,s):
    cout.info("Interpolating total ATMF variance")
    if len(k)>2:
        f = spline(k,s,k=2)
    else:
        f = spline(k,s,k=1)
    atm = (float(f(0))**2)*(T/base)
    cout.info(f"{round(T[0])} ATMF Variance {atm[0]}")
    return atm

def eed(C,P,S,K,r,q,T):
    return C-P-S*exp(-q*T)+K*exp(-r*T)
#---------------------------------------------------------------------------------------------
