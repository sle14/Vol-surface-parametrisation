from scipy.optimize import curve_fit,minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import log,sqrt,exp,inf,pi
import pandas as pd
import scipy as sp
import numpy as np
import sqlalchemy
import math
cdf,pdf = norm.cdf,norm.pdf
pd.options.mode.chained_assignment = None
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')

def get_vega(X,T,S,d1,r,q):
    T = T/365
    return (S*exp(-q*T)*pdf(d1)*sqrt(T))/100

def get_ds(X,T,S,r,q,s): 
    T = T/365
    d1 = (log(S/X)+(r-q+(s**2)/2)*T)/(s*sqrt(T))
    d2 = d1-(s*sqrt(T))
    return d1,d2

def get_tenor(df,cols):
    for x in cols: df[x] = pd.to_datetime(df[x],format="%d/%m/%Y")
    df["Tenor"] = (df[cols[1]]-df[cols[0]]).dt.days
    for x in cols: df[x] = df[x].apply(lambda x:x.strftime("%d/%m/%Y"))
    return df

def get_frame(database,query):
    server="LAPTOP-206OR7PL\\SQLEXPRESS"
    engine = sqlalchemy.create_engine('mssql+pyodbc://@'+server+'/'+
    database+'?trusted_connection=yes&driver=ODBC+Driver+13+for+SQL+Server')
    return pd.read_sql(query,con=engine)

def get_value(R,S,X,T,r,q,s): 
    T = T/365
    d1 = (log(S/X)+(r-q+(s**2)/2)*T)/(s*sqrt(T))
    d2 = d1-(s*sqrt(T))
    if R == "Call":
        return (S*exp(-q*T)*cdf(d1))-(X*exp(-r*T)*cdf(d2))
    else:
        return (X*exp(-r*T)*cdf(-d2))-(S*exp(-q*T)*cdf(-d1))
    
def bisect(R,P,S,X,T,r,q):
    precision,max_iterations = 1e-3,500
    upper_vol,lower_vol = 1000,0.01
    error,i = 1,0
    while error > precision and i < max_iterations:
        i += 1
        mid_vol = (upper_vol+lower_vol)/2
        price = get_value(R,S,X,T,r,q,mid_vol)
        error = abs(price-P)
        if R == "Call":
            lower_price = get_value(R,S,X,T,r,q,lower_vol)
            if (lower_price - P) * (price - P) > 0:
                lower_vol = mid_vol
            else:
                upper_vol = mid_vol
        else:
            upper_price = get_value(R,S,X,T,r,q,upper_vol)
            if (upper_price - P) * (price - P) > 0:
                upper_vol = mid_vol
            else:
                lower_vol = mid_vol
    if mid_vol < 500: 
        if mid_vol < 0.02:
            return np.nan
        else:
            return mid_vol
    else:
        return np.nan

def get_forward(T,S,r,q): return S*exp((-r-q)*(T/365))
    
def get_logstrike(X,F): return log(X/F)

class SVI(object):
    def __init__(self,df):
        self.chi = pd.DataFrame(columns=["t","q","epsilon","a","b","rho","m","sigma"]) 
        self.df0 = pd.DataFrame(columns=df.columns)
        self.df1 = pd.DataFrame(columns=df.columns)
        self.arbfree = []
        self.arbfit = []
        self.errors = []
        self.t0,self.t1 = int,int
        self.df = df #Whole grid
        self.q = [0]

    def risk_neutral_density(self,x,a,b,rho,m,sigma):
        w0 = self.raw(x,a,b,rho,m,sigma)
        w1 = b*(rho+(x-m)/sqrt((x-m)**2+sigma**2))
        w2 = b*((sqrt((x-m)**2+sigma**2)-(x-m)**2/sqrt((x-m)**2+sigma**2))/((x-m)**2+sigma**2))
        g = (1-x*w1/(2*w0))**2-w1**2/4*(1/w0+1/4)+w2/2
        d = -x/sqrt(w0)-sqrt(w0)/2
        return g/sqrt(2*pi*w0)*exp((-d**2)/2)

    def raw(self,x,a,b,rho,m,sigma): 
        return a+b*(rho*(x-m)+sqrt((x-m)**2+sigma**2))

    def raw2wing(self,a,b,rho,m,sigma,t):
        t = t/365
        v = (a+b*(-rho*m+sqrt((m**2)+(sigma**2))))/t
        w = v*t
        psi = 1/sqrt(w)*b/2*(-m/sqrt(m**2+sigma**2)+rho)
        p = 1/sqrt(w)*b*(1-rho)
        c = 1/sqrt(w)*b*(1+rho)
        u = 1/t*(a+b+sigma*sqrt(1-rho**2))
        return pd.DataFrame({"v":v,"w":w,"psi":psi,"p":p,"c":c,"u":u})

    def div_residual(self,q,x,y):
        df = self.df0
        bid = df["Bid"].to_numpy()
        ask = df["Ask"].to_numpy()
        mid = np.vectorize(get_value)(df["Type"],df["Spot"],x,df["Tenor"],df["Rate"],q,df["ImpliedVolSvi"])
        mid_bid = ((mid-bid)/((mid-bid)+(ask-mid)))**2
        ask_mid = ((ask-mid)/((mid-bid)+(ask-mid)))**2
        epsilon = sum(((mid_bid+ask_mid)/.5)-1)*100
        return epsilon 

    def var_residual(self,params,x,y):
        df = self.df0
        vega = df["Vega"]
        weights = [i/max(vega) for i in vega]
        #Quadratically weighted penalty for prices derived from the fit being outside the spread
        bid = df["Bid"].to_numpy()
        ask = df["Ask"].to_numpy()
        var = self.raw(x,*params)
        vol = sqrt(var)
        mid = np.vectorize(get_value)(df["Type"],df["Spot"],df["Strike"],df["Tenor"],df["Rate"],self.q,vol)
        mid_bid = ((mid-bid)/((mid-bid)+(ask-mid)))**2
        ask_mid = ((ask-mid)/((mid-bid)+(ask-mid)))**2
        outspread = sum(((mid_bid+ask_mid)/.5)-1)  
        butterfly = abs(weights*(y-var)).sum()
        if self.t1 != 365: #Arbitrary number for expiry outside our final slice
            var_next = self.df1["ImpliedVar"].to_numpy()
            calendar = sum([np.maximum(0,x1-x0) for x0,x1 in zip(var,var_next)])*10000 #Heavy penalty on calendars
            epsilon = (butterfly+outspread+calendar)/3 #Mean residual
        else:
            epsilon = (butterfly+outspread)/3
        self.chi.loc[len(self.chi)] = [self.t0,self.q[0],epsilon]+params.tolist()
        self.errors.append(epsilon)
        return epsilon 
    
    def optimise(self):
        epsilon,tolerance = inf,2
        tenors0 = sorted(set(self.df["Tenor"].unique()))
        tenors1 = sorted(set(np.append(self.df["Tenor"].unique()[1:],365)))
        for t0,t1 in zip(tenors0,tenors1):
            self.t0 = t0
            self.t1 = t1
            while epsilon > tolerance:
                self.var_fit()
                epsilon = self.arbfree[2]
                print(f"q {self.q[0]:.3f} | t0 slice {t0} | t1 slice {t1} | ε {epsilon:.3f}")
                if epsilon > tolerance: self.div_fit() #If error is too big we are missig q
            epsilon = inf
            self.graph()
            self.errors = []
        self.chi = self.chi[self.chi["epsilon"]==self.chi.groupby("t")["epsilon"].transform("min")]   
        return self.chi.join(self.raw2wing(self.chi["a"],self.chi["b"],self.chi["rho"],self.chi["m"],self.chi["sigma"],self.chi["t"]))
    
    def div_fit(self):
        x0 = self.df0["Strike"].to_numpy()
        y0 = self.df0["Mid"].to_numpy()
        self.q = minimize(self.div_residual,(self.q),method="SLSQP",args=(x0,y0),options={"maxiter":100}).x
        return self
    
    def prepare_grid(self,t,q):
        df = self.df[self.df["Tenor"]==t]
        df["ImpliedVol"] = np.vectorize(bisect)(df["Type"],df["Mid"],df["Spot"],df["Strike"],df["Tenor"],df["Rate"],q)
        df["ImpliedVar"] = df["ImpliedVol"]**2
        df["Forward"] = np.vectorize(get_forward)(df["Tenor"],df["Spot"],df["Rate"],q)
        df["LogStrike"] = np.vectorize(get_logstrike)(df["Strike"],df["Forward"])
        df["d1"],df["d2"] = np.vectorize(get_ds)(df["Strike"],df["Tenor"],df["Spot"],df["Rate"],q,df["ImpliedVol"])
        df["Vega"] = np.vectorize(get_vega)(df["Strike"],df["Tenor"],df["Spot"],df["d1"],df["Rate"],q)
        df = df.dropna().sort_values(by=["LogStrike"])
        
        xdata = df["LogStrike"].to_numpy()
        ydata = df["ImpliedVar"].to_numpy()
        a_ = (0,max(ydata)) #vertical translation
        b_ = (0,10) #wings slope
        rho_ = (-1,1) #counter-clockwise rotation
        m_ = (2*min(xdata),2*max(xdata)) #horizontal translation
        sigma_ = (0.01,10) #smile curvature

        #Get fit for smile as first guess of params 
        lower_bound = (a_[0],b_[0],rho_[0],m_[0],sigma_[0])
        upper_bound = (a_[1],b_[1],rho_[1],m_[1],sigma_[1])
        self.arbfit,pcov = curve_fit(self.raw,xdata,ydata,bounds=[lower_bound,upper_bound])
        return xdata,ydata,(a_,b_,rho_,m_,sigma_),df
    
    def var_fit(self):
        if self.t1 != 365: x1,y1,bounds1,self.df1 = self.prepare_grid(self.t1,0)
        x0,y0,bounds0,self.df0 = self.prepare_grid(self.t0,self.q)
        #Imposing constraints (0<x<4 and 1<x) so that RND's P(x)>0 making smile free of fly arb
        def left_wing_first_coefficient_lhs(x): return ((x[1]**2)*(x[2]+1)**2)-0
        def left_wing_first_coefficient_rhs(x): return ((x[1]**2)*(x[2]+1)**2)+4
        def right_wing_first_coefficient_lhs(x): return ((x[1]**2)*(x[2]-1)**2)-0
        def right_wing_first_coefficient_rhs(x): return ((x[1]**2)*(x[2]-1)**2)+4
        def left_wing_discriminant(x): return (x[0]-x[3]*x[1]*(x[2]-1))*(4-x[0]+x[3]*x[1]*(x[2]-1))-(x[1]**2*(x[2]-1)**2)-0
        def right_wing_discriminant(x): return (x[0]-x[3]*x[1]*(x[2]+1))*(4-x[0]+x[3]*x[1]*(x[2]+1))-(x[1]**2*(x[2]+1)**2)-0

        #Minimise implied variance error between fit and arbfree smile
        guess = (0.5*min(y0),0.1,-0.5,0.1,0.1)
        minimize(self.var_residual,guess,method="SLSQP",args=(x0,y0),options={"maxiter":1000},
                 bounds=bounds0,
                 constraints=({"type":"ineq","fun":left_wing_first_coefficient_lhs},
                              {"type":"ineq","fun":left_wing_first_coefficient_rhs},
                              {"type":"ineq","fun":right_wing_first_coefficient_lhs},
                              {"type":"ineq","fun":right_wing_first_coefficient_rhs},
                              {"type":"ineq","fun":left_wing_discriminant},
                              {"type":"ineq","fun":right_wing_discriminant}))
        
        self.arbfree = self.chi[self.chi["t"]==self.t0]
        self.arbfree = self.arbfree[self.arbfree["epsilon"]==min(self.arbfree["epsilon"])].values.tolist()[0]
        
        #New variance,volatility and mid prices
        self.df0["ImpliedVarSvi"] = self.raw(x0,*self.arbfree[3:])
        self.df0["ImpliedVolSvi"] = sqrt(self.df0["ImpliedVarSvi"])
        self.df0["MidSvi"] = np.vectorize(get_value)(self.df0["Type"],self.df0["Spot"],self.df0["Strike"],self.df0["Tenor"],self.df0["Rate"],0,self.df0["ImpliedVolSvi"])
        self.df0["MidSviQ"] = np.vectorize(get_value)(self.df0["Type"],self.df0["Spot"],self.df0["Strike"],self.df0["Tenor"],self.df0["Rate"],self.q,self.df0["ImpliedVolSvi"])
        return self

    def graph(self):
        df = self.df0
        t = str(self.t0)
        logstrike = df["LogStrike"].to_numpy()
        impliedvar = df["ImpliedVar"].to_numpy()
        #Graph errors
        plt.scatter(range(len(self.errors)),self.errors,s=1,label="Epsilon "+t)
        plt.yscale("log")
        plt.legend()
        plt.show()
        #Graph risk neutral density of best fit and arb free smile
        plt.plot(logstrike,self.risk_neutral_density(logstrike,*self.arbfree[3:]),'g',label="Arb-free "+t)
        plt.plot(logstrike,self.risk_neutral_density(logstrike,*self.arbfit),'r',label="Best-fit "+t)
        plt.legend()
        plt.show()
        #Graph implied variance for raw values, best fit and arb free smile
        plt.plot(logstrike,impliedvar,label="Raw "+t)
        plt.plot(logstrike,self.raw(logstrike,*self.arbfree[3:]),'g',label="Arb-free "+t)
        plt.plot(logstrike,self.raw(logstrike,*self.arbfit),'r',label="Best-fit "+t)
        plt.legend()
        plt.show()
        df = df[(df["Moneyness"]>=0.9)&(df["Moneyness"]<=1.1)]
        #Plot Prices - Call
        plt.plot(df[df["Type"]=="Call"]["Moneyness"],df[df["Type"]=="Call"]["Bid"].astype(float),'r',label="Call Bid "+t)
        plt.plot(df[df["Type"]=="Call"]["Moneyness"],df[df["Type"]=="Call"]["Ask"].astype(float),'g',label="Call Ask "+t)
        plt.plot(df[df["Type"]=="Call"]["Moneyness"],df[df["Type"]=="Call"]["MidSvi"],'y',label="Call Svi "+t)
        plt.plot(df[df["Type"]=="Call"]["Moneyness"],df[df["Type"]=="Call"]["MidSviQ"],'b',label="Call SviQ "+t)
        plt.legend()
        plt.show()   
        #Plot Prices - Put
        plt.plot(df[df["Type"]=="Put"]["Moneyness"],df[df["Type"]=="Put"]["Bid"].astype(float),'r',label="Put Bid "+t)
        plt.plot(df[df["Type"]=="Put"]["Moneyness"],df[df["Type"]=="Put"]["Ask"].astype(float),'g',label="Put Ask "+t)
        plt.plot(df[df["Type"]=="Put"]["Moneyness"],df[df["Type"]=="Put"]["MidSvi"],'y',label="Put Svi "+t)
        plt.plot(df[df["Type"]=="Put"]["Moneyness"],df[df["Type"]=="Put"]["MidSviQ"],'b',label="Put SviQ "+t)    
        plt.legend()
        plt.show() 

def trim_grid(df):
    df = get_tenor(df,["Date","Expiry"])
    df["Bid"] = df["Bid"].astype(float)
    df["Ask"] = df["Ask"].astype(float)
    df["Spread"] = df["Spread"].astype(float)
    df["Moneyness"] = df["Strike"]/df["Spot"]
    df = df[(df["Mid"]!=0)&(df["Tenor"]!=0)&(df["Bid"]>0)&(df["Ask"]>0)&(df["Moneyness"]>.7)&(df["Moneyness"]<1.2)]
    return df

#and dbo.options.Expiry = '18/09/2020'
query = """
select dbo.options.*, dbo.equities.Mid as 'Spot', dbo.rates.Rate
from dbo.options 
join dbo.equities on dbo.options.Date = dbo.equities.Date and dbo.options.Date = '15/04/2020'
join dbo.rates on dbo.options.Date = dbo.rates.Date"""
df = trim_grid(get_frame("Historical",query))
#--------------SVI----------------------------------------------------------------------------
chi = SVI(df).optimise()



