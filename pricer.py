from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import curve_fit,bisect,newton
from scipy.stats import norm
from scipy import stats
from numpy import log,sqrt,exp,inf
from numpy import vectorize as vct
import numpy as np
import pandas as pd
import time
N,n = norm.cdf,norm.pdf
start_time = time.time()
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')

def simps(f,x,y,dt):
    y = np.array(y)
    t = dt*np.flip(np.array(range(len(y))))
    I = np.nan_to_num(f(x,y,t))
    I[1:-1] = 2*I[1:-1]
    idx = list(range(1,len(y)-2,2))
    I[idx] = 2*I[idx]
    return dt/3*I.sum()

def trapz(f,x,y,dt):
    y = np.array(y)
    t = dt*np.flip(np.array(range(len(y))))
    I = dt*np.nan_to_num(f(x,y,t))
    I[[0,-1]] = I[[0,-1]]/2    
    return I.sum()

class KimIntegral:
    def __init__(self,R,M,T,S,K,s,r,q,H=None):
        self.q,self.T,self.dt = q,T,T/M
        self.K,self.s,self.r = K,s,r
        self.R,self.S,self.M = R,S,M
        #Halving coefficients from spread fit on american quotes
        if H != None:
            self.H0,self.H1,self.Kh = np.array(H)[0]/2,np.array(H)[1]/2,H[2]
        #Terminal boundaries to ensure smooth pasting conditions
        if self.R == "C" and M > 1:
            self.Bm = float(np.where((q>0)&(r>0),K*np.maximum(1,r/q),K))
        elif M > 1:
            self.Bm = float(np.where((q>0)&(r>0),K*np.minimum(1,r/q),K))

    #Half spread function
    def H(self,x):
        H0,H1,Kh = self.H0,self.H1,self.Kh
        if self.R == "C":
            return H0 + H1*np.maximum(x-Kh,0)
        else:
            return H0 + H1*np.maximum(Kh-x,0)
    
    #Minimise difference between payoff from selling vs exercising 
    def residual(self,Bi,i,B:list):
        if Bi < 0: return 1e100
        H,K,dt = self.H,self.K,self.dt
        eep = trapz(self.f,Bi,B+[Bi],dt)
        eur = self.v(Bi,K,(i*dt))
        epsilon = K - Bi + eur + eep - H(Bi) if self.R == "C" else Bi - K + eur + eep - H(Bi)
        if abs(epsilon) < (self.T/1e2)**2: 
            return 0
        else:
            return epsilon if np.isnan(epsilon) == False else 1e6 
    
    #Find boundary through backward induction starting at terminal boundary on expiry
    def fit_boundary(self):
        Bm,T,M,dt = self.Bm,self.T,self.M,self.dt
        Bi,B,I = Bm,[Bm],[T]#;print(0,Bm)
        for i in range(1,M+1):
            try:
                Bi = newton(self.residual,x0=Bm,args=(i,B))
            except:
                a = Bm/10 if self.R == "C" else 0
                b = Bm*10 if self.R == "C" else Bm
                Bi = bisect(self.residual,a,b,args=(i,B))
            B.append(Bi);I.append(T-i*dt)#;print(i,Bi)
        return B,I

    def d1(self,x,y,t):
        s,r,q = self.s,self.r,self.q
        return (log(x/y)+(r-q+(s**2)/2)*t)/(s*sqrt(t))
    
    def d2(self,x,y,t):
        s,r,q = self.s,self.r,self.q
        return (log(x/y)+(r-q-(s**2)/2)*t)/(s*sqrt(t))
        
    #Early exercise premium
    def f(self,x,y,t):
        K,r,q,d1,d2 = self.K,self.r,self.q,self.d1,self.d2
        if self.R == "C":
            return np.where(t!=0, (q*x*exp(-q*t)*N(d1(x,y,t)))-(r*K*exp(-r*t)*N(d2(x,y,t))),
                   np.where(x<y,   q*x-r*K,
                   np.where(x>y,   0, 
                                   1/2*(q*x-r*K) )))
        else:
            return np.where(t!=0, (r*K*exp(-r*t)*N(-d2(x,y,t)))-(q*x*exp(-q*t)*N(-d1(x,y,t))),
                   np.where(x<y,   r*K-q*x,
                   np.where(x>y,   0, 
                                   1/2*(r*K-q*x) )))  
    #BSM european value
    def v(self,x=None,y=None,t=None):
        r,q,d1,d2 = self.r,self.q,self.d1,self.d2
        if x==None and y==None and t==None: x,y,t = self.S,self.K,self.T
        if self.R == "C":
            return (x*exp(-q*t)*N(d1(x,y,t)))-(y*exp(-r*t)*N(d2(x,y,t)))
        else:
            return (y*exp(-r*t)*N(-d2(x,y,t)))-(x*exp(-q*t)*N(-d1(x,y,t)))
    
    #Get european value and integrate boundaries to get early exercise premium
    def get_value(self,B):
        T,K,S,dt = self.T,self.K,self.S,self.T/(len(B)-1)
        eur = self.v(S,K,T) 
        eep = simps(self.f,S,B,dt) 
        return eur,eep
#---------------------------------------------------------------------------------------------
#Back out implied divs and borrows from quotes
def impdivsborrows(df,call_col,put_col,trim_pct):
    def f(S,K,C,P,T,r,D,Td): return -1/(T/365)*log((C-P+K*exp(-r*(T/365))+D*exp(-r*(Td/365)))/S)
    df["ImpBorrow"] = vct(f)(df["MidSpot"],df["Strike"],df[call_col],df[put_col],df["Tenor"],df["Rate"],df["Div"],df["CumDivDays"])
    df["ImpBorrowDiv"] = vct(f)(df["MidSpot"],df["Strike"],df[call_col],df[put_col],df["Tenor"],df["Rate"],0,0)
    def trim(x): return stats.trim_mean(x,trim_pct) #Truncated mean
    df["MeanImpBorrow"] = df.groupby(["Expiry","Time"])["ImpBorrow"].transform(trim)
    df["MeanImpBorrowDiv"] = df.groupby(["Expiry","Time"])["ImpBorrowDiv"].transform(trim)
    return df

#Interpolate cols
def interp(df,*cols):
    df = df.sort_values(by=["Tenor","Time","Strike"])
    for T in sorted(df["Tenor"].unique()):
        for t  in sorted(df["Time"].unique()):
            for c in cols:
                df.loc[(df["Tenor"]==T)&(df["Time"]==t),c] = df.loc[(df["Tenor"]==T)&(df["Time"]==t),c].interpolate(
                    method="spline",order=2,limit_direction="backward",limit=1)
    return df

#Fit spread function on quotes to get params
def fit_spread(df,R,ref_bid,ref_ask,new_col):
    params = pd.DataFrame(columns=["Tenor","Time",new_col])
    def f(x,H0,H1,Kh):
        return H0 + H1*np.maximum(Kh-x,0) if R == "C" else H0 + H1*np.maximum(x-Kh,0)
    for T in sorted(df["Tenor"].unique()):
        for t  in sorted(df["Time"].unique()):
            dfx = df[(df["Tenor"]==T)&(df["Time"]==t)]
            K = dfx["Strike"]#; print([T,t])
            spread = dfx[ref_ask]-dfx[ref_bid]
            popt,pcov = curve_fit(f,K,spread,bounds=((0),(inf)),p0=(min(spread),0.01,df["MidSpot"].iloc[0]))
            params.loc[len(params)] = [T,t,list(popt)]
    return df.set_index(["Tenor","Time"]).join(params.set_index(["Tenor","Time"])).reset_index()

#Get forward with div as discrete
def forward(T,S,r,q,Td,D):
    T,Td = T/365,Td/365
    return (S-D*exp(-r*Td))*exp((r-q)*T)

#Interpolate total atm variance
def tot_atm_var(df,k_col,vol_col,new_col):
    atm_vars = pd.DataFrame(columns=["Tenor","Time",new_col])
    for t in sorted(df["Time"].unique()):
        for T in sorted(df["Tenor"].unique()):
            dfx = df[(df["Tenor"]==T)&(df["Time"]==t)].sort_values(k_col)
            x,y = dfx[k_col],dfx[vol_col]
            f = spline(x,y,k=2)
            atm_var = (float(f(0))**2)*(T/365)
            atm_vars.loc[len(atm_vars)] = [T,t,atm_var] #;print([t,atm_var]) 
    return df.set_index(["Tenor","Time"]).join(atm_vars.set_index(["Tenor","Time"])).reset_index() 

#Get variances and total variances 
def tot_var(s,T):
    return s**2*(T/365),s**2
        
#Solve vol
class Vol:
    def __init__(self,AE,R,M=None):
        self.R,self.M,self.AE = R,M,AE
        self.eep,self.eeb = 0,None

    def residual_eur(self,s,PX,T,S,K,r,q,t): 
        epsilon = PX - KimIntegral(self.R,1,T/365,S,K,s,r,q).v()
        if abs(epsilon) > 1e-5:
            return epsilon
        else:
            #print([self.AE,self.R,T,t,K,s])
            return 0 
        
    def residual_ame(self,s,PX,T,S,K,r,q,H,t):
        if s < 0: return 1e6
        KI = KimIntegral(self.R,self.M,T/365,S,K,s,r,q,H)
        B,I = KI.fit_boundary()
        eur,eep = KI.get_value(B)
        epsilon = PX - (eur + eep)
        if abs(epsilon) > 1e-5:
            return epsilon
        else:
            self.eep,self.eeb = eep,B#; print([self.AE,self.R,T,t,K,s])
            return 0   
            
    def root(self,PX,T,S,K,r,q,t,x0=None,H=None):
        if self.AE == "A":
            try:
                s = newton(self.residual_ame,x0=x0,args=(PX,T,S,K,r,q,H,t))
                return s,self.eep,np.array(self.eeb,dtype=object)
            except:
                try:
                    s = bisect(self.residual_ame,0.1,2,args=(PX,T,S,K,r,q,H,t))
                    return s,self.eep,np.array(self.eeb,dtype=object)
                except:
                    #print("No solution for",[self.AE,self.R,T,t,K])
                    return np.nan,np.nan,np.array([0]*(self.M+1),dtype=object)
        else:
            try:
                return bisect(self.residual_eur,0.1,2,args=(PX,T,S,K,r,q,t))
            except:
                try:
                    return newton(self.residual_eur,x0=1,args=(PX,T,S,K,r,q,t))
                except:
                    #print("No solution for",[self.AE,self.R,T,t,K])
                    return np.nan