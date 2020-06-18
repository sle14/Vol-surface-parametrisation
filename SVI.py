from scipy.optimize import curve_fit,minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import log,sqrt,exp,inf,pi
import pandas as pd
import numpy as np
import sqlalchemy
import time
cdf,pdf = norm.cdf,norm.pdf
pd.options.mode.chained_assignment = None
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')

def sign(i): return -1 if i < 0 else 1

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

def get_implied_div(df,ex_itm_puts=False,plot=False):
    dfx = df[df["Moneyness"]>1] if ex_itm_puts==True else df
    S,r = df["Spot"].iloc[0],df["Rate"].iloc[0]
    def q(K,S,CP,T,r): return -1/(T/365)*log((CP+K*exp(-r*(T/365)))/S)   
    dfx = dfx[["Date","Tenor","Strike","Type","Mid"]].set_index(["Date","Tenor","Strike","Type"]).unstack(3)
    dfx[("PutCallSpread","PutCallSpread")] = dfx[("Mid","Call")] - dfx[("Mid","Put")]
    dfx = dfx[[("PutCallSpread","PutCallSpread")]].stack(1).reset_index().dropna(); del dfx["Type"]
    dfx["ImpliedDiv"] = np.vectorize(q)(dfx["Strike"],S,dfx["PutCallSpread"],dfx["Tenor"],r)
    if plot==True:
        for t in sorted(dfx["Tenor"].unique()):
            plt.plot(dfx[dfx["Tenor"]==t]["Strike"],dfx[dfx["Tenor"]==t]["ImpliedDiv"],label=str(t))
        plt.legend(),plt.show()
    return df.set_index("Tenor").join(dfx.groupby(["Tenor"]).mean()[["ImpliedDiv"]]).reset_index()
   
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
    
def get_strike_vector(minK,dK,maxK): return np.array([i*dK for i in range(int(minK/dK),int(maxK/dK)+3)])

def get_logstrike(K,T,S,r,q): return log(K/(S*exp((-r-q)*(T/365))))

def get_logstrike_vector(mink,dk,maxk): return np.array([i*dk for i in range(int(mink/dk),int(maxk/dk)+1)])

#Gatheral density: taking first and second derivative of variance to logstrike
def rnd(k,a,b,rho,m,sigma):
    w0 = raw(k,a,b,rho,m,sigma)
    w1 = b*(rho+((k-m)/sqrt((k-m)**2+sigma**2)))
    w2 = (b*sigma**2)/((k-m)**2+sigma**2)**(3/2)
    g = ((1-((k*w1)/(2*w0)))**2)-((w1**2)/4)*((1/w0)+0.25)+(w2/2)
    p = (g/(w0*sqrt(2*pi))*exp(-(k/sqrt(w0)+sqrt(w0)/2)**2))
    return p

#Breeden&Litzenberger density: taking second derivative of call price to strike
def rnd2(K,T,S,r,q,a,b,rho,m,sigma):
    dK = (K[1]-K[0])
    k = get_logstrike(K,T,S,r,q)
    C = get_value("Call",S,K,T,r,q,sqrt(raw(k,a,b,rho,m,sigma)))
    Q = [exp(r*(T/365))*(c2-(2*c1)+c0)/dK for c2,c1,c0 in zip(C[2:],C[1:-1],C[:-2])]
    return Q

def draw_graphs(df,params,S,T,r):
    q = df["ImpliedDiv"].iloc[0]
    df["ImpliedVarSvi"] = raw(df["LogStrike"].to_numpy(),*params)
    df["ImpliedVolSvi"] = sqrt(df["ImpliedVarSvi"])
    df["MidSvi"] = np.vectorize(get_value)(df["Type"],S,df["Strike"],T,r,0,df["ImpliedVolSvi"])
    df["MidSviDiv"] = np.vectorize(get_value)(df["Type"],S,df["Strike"],T,r,q,df["ImpliedVolSvi"])
    df["MidSpread"] = df["Spread"]/2
    df["MidError"] = abs(df["Mid"]-df["MidSvi"])
    df["MidErrorDiv"] = abs(df["Mid"]-df["MidSviDiv"])
    
    K = get_strike_vector(20,0.01,50)
    k = get_logstrike(K,T,S,r,q)
    w = raw(k,*params)

    plt.title("Risk-neutral density: slice "+str(int(T)))
    plt.plot(k,rnd(k,*params),c="r",label="Density",linewidth=1)
    plt.legend(); plt.show()
    
    plt.title("Implied variance: slice "+str(int(T)))
    plt.scatter(df["LogStrike"],df["ImpliedVar"],s=2,c="k",label="BSM")
    plt.plot(k,w,c="r",label="SVI",linewidth=1)
    plt.legend(); plt.show()
    
    plt.title("Put errors: slice "+str(int(T))); df_put = df[df["Type"]=="Put"]
    plt.plot(df_put["Strike"],df_put["MidErrorDiv"].astype(float),c='r',linewidth=1)
    plt.plot(df_put["Strike"],df_put["MidError"].astype(float),c='y',linewidth=1)
    plt.plot(df_put["Strike"],df_put["MidSpread"].astype(float),c='k',linewidth=1)
    plt.legend(); plt.show()  

    plt.title("Call errors: slice "+str(int(T))); df_call = df[df["Type"]=="Call"]
    plt.plot(df_call["Strike"],df_call["MidErrorDiv"].astype(float),c='r',linewidth=1)
    plt.plot(df_call["Strike"],df_call["MidError"].astype(float),c='y',linewidth=1)
    plt.plot(df_call["Strike"],df_call["MidSpread"].astype(float),c='k',linewidth=1)
    plt.legend(); plt.show()   

def raw(x,a,b,rho,m,sigma): return a+b*(rho*(x-m)+sqrt((x-m)**2+sigma**2))

#Imposing constraints (0<x<4 and 1<x) so that RND's P(x)>0 making smile free of fly arb  
def left_wing_first_coefficient_lhs(x): return ((x[1]**2)*(x[2]+1)**2)-0
def left_wing_first_coefficient_rhs(x): return ((x[1]**2)*(x[2]+1)**2)+4
def right_wing_first_coefficient_lhs(x): return ((x[1]**2)*(x[2]-1)**2)-0
def right_wing_first_coefficient_rhs(x): return ((x[1]**2)*(x[2]-1)**2)+4
def left_wing_discriminant(x): return (x[0]-x[3]*x[1]*(x[2]-1))*(4-x[0]+x[3]*x[1]*(x[2]-1))-(x[1]**2*(x[2]-1)**2)-0
def right_wing_discriminant(x): return (x[0]-x[3]*x[1]*(x[2]+1))*(4-x[0]+x[3]*x[1]*(x[2]+1))-(x[1]**2*(x[2]+1)**2)-0
def slope_rhs(x,T): return (x[1])+(4/((T/365)*(1+abs(x[2]))))
def slope_lhs(x): return (x[1])-0

def raw2wings(a,b,rho,m,sigma,t):
    t = t/365
    theta = (a+b*(-rho*m+sqrt((m**2)+(sigma**2))))
    psi = 1/sqrt(theta)*b/2*(-m/sqrt(m**2+sigma**2)+rho)
    p = 1/sqrt(theta)*b*(1-rho)
    c = 1/sqrt(theta)*b*(1+rho)
    u = 1/t*(a+b*sigma*sqrt(1-rho**2))
    return theta,psi,p,c,u

def wings2raw(theta,psi,p,c,u,t):
    t = t/365
    b = sqrt(theta)/2*(c+p)
    rho = 1-p*sqrt(theta)/b
    beta = rho-2*psi*sqrt(theta)/b
    alpha = sign(beta)*sqrt(1/beta**2-1)
    m = t*((theta/t)-u)/(b*(-rho+sign(alpha)*sqrt(1+alpha**2)-alpha*sqrt(1-rho**2)))
    sigma = alpha*m if m != 0 else ((u*t-theta)/b)/sqrt(1-rho**2)
    sigma = 0 if sigma < 0 else sigma
    a = u*t-b*sigma*sqrt(1-rho**2)
    return a,b,rho,m,sigma
    
class SVI(object):
    def __init__(self,df):
        self.chi = pd.DataFrame(columns=["t","q","epsilon","a","b","rho","m","sigma"]) 
        self.slices = sorted(set(df["Tenor"].unique()))
        self.S = df["Spot"].iloc[0]
        self.r = df["Rate"].iloc[0]
        self.df = df

    def residual(self,params,df,i,K):
        params = params.tolist()
        t,q = df["Tenor"].iloc[0],df["ImpliedDiv"].iloc[0]
        bid,ask,var = df["Bid"].to_numpy(),df["Ask"].to_numpy(),raw(df["LogStrike"].to_numpy(),*params)
        mid_svi = np.vectorize(get_value)(df["Type"],self.S,df["Strike"],self.slices[i],self.r,df["ImpliedDiv"],sqrt(var))
        mid_bid = ((mid_svi-bid)/((mid_svi-bid)+(ask-mid_svi)))**2
        ask_mid = ((ask-mid_svi)/((mid_svi-bid)+(ask-mid_svi)))**2
        spread = (((mid_bid+ask_mid)/.5)-1)
        weights = [1-((i-self.S)/self.S)**2 for i in df["Strike"]]
        epsilon = sum([s for s,w in zip(spread,weights)])
        self.chi.loc[len(self.chi)] = [t,q,epsilon]+params
        self.chi = self.chi[self.chi["epsilon"]==self.chi.groupby("t")["epsilon"].transform("min")]
        self.chi = self.chi.drop_duplicates(subset=["t"]).reset_index(drop=True).sort_values(by=["t"])
        if np.isnan(epsilon) == True:
            return 1e4
        else:
            return epsilon
    
    def work_frame(self,i):
        t = self.slices[i]
        df = self.df[self.df["Tenor"]==t]
        df["ImpliedVol"] = np.vectorize(bisect)(df["Type"],df["Mid"],df["Spot"],df["Strike"],df["Tenor"],df["Rate"],df["ImpliedDiv"])
        df["ImpliedVar"] = (df["ImpliedVol"]**2)
        df["LogStrike"] = np.vectorize(get_logstrike)(df["Strike"],df["Tenor"],df["Spot"],df["Rate"],df["ImpliedDiv"])
        df = df.dropna().sort_values(by=["LogStrike"])
        return df

    def get_params(self,i):
        t = self.slices[i]
        chi = list(self.chi[self.chi["t"]==t].iloc[0])
        epsilon,params = chi[2],chi[3:]
        return epsilon,params

    def optimise(self,i,epsilon,tolerance,K,graphs:bool):
        t = self.slices[i]          
        while epsilon > tolerance:
            df = self.work_frame(i)
            self.px_fit(df,i,K)
            epsilon,params = self.get_params(i)
        if graphs==True: draw_graphs(df,params,self.S,t,self.r) 

    def run(self):
        epsilon,tolerance = inf,10
        K = get_strike_vector(20,1,60)
        for i in range(len(self.slices)):
            self.optimise(i,epsilon,tolerance,K,True)
        chi = self.chi
        jw = raw2wings(chi["a"],chi["b"],chi["rho"],chi["m"],chi["sigma"],chi["t"])
        return chi.join(pd.DataFrame({"w":jw[0],"psi":jw[1],"p":jw[2],"c":jw[3],"u":jw[4],"r":self.r,"S":self.S}))

    def px_fit(self,df,i,K):
        k = df["LogStrike"].to_numpy()
        w = df["ImpliedVar"].to_numpy()
        _a,_b,_rho,_m,_sigma = 0,0,-1,2*min(k),0.01
        a_,b_,rho_,m_,sigma_ = max(w),10,1,2*max(k),10
        args = (df,i,K)
        bounds = ((_a,a_),(_b,b_),(_rho,rho_),(_m,m_),(_sigma,sigma_))
        constraints = ({"type":"ineq","fun":left_wing_first_coefficient_lhs},
                       {"type":"ineq","fun":left_wing_first_coefficient_rhs},
                       {"type":"ineq","fun":right_wing_first_coefficient_lhs},
                       {"type":"ineq","fun":right_wing_first_coefficient_rhs},
                       {"type":"ineq","fun":left_wing_discriminant},
                       {"type":"ineq","fun":right_wing_discriminant},
                       {"type":"ineq","fun":slope_rhs,"args":[self.slices[i],]},
                       {"type":"ineq","fun":slope_lhs})
        lower_bound = _a,_b,_rho,_m,_sigma
        upper_bound = a_,b_,rho_,m_,sigma_
        guess = curve_fit(raw,k,w,bounds=[lower_bound,upper_bound])[0]
        minimize(self.residual,guess,method="SLSQP",args=args,options={"maxiter":1000},bounds=bounds,constraints=constraints)
        print(f"t {self.slices[i]:={3}} ε {self.get_params(i)[0]:.3f}")
        
def trim_grid(df):
    df = get_tenor(df,["Date","Expiry"])
    cols = ["Strike","Bid","Ask","Spread","Mid"]
    df[cols] = df[cols].astype(float)
    df["Moneyness"] = df["Strike"]/df["Spot"]
    df = df[(df["Tenor"]!=0)&(df["Bid"]>0.1)&(df["Ask"]>0.1)]
    return df

query = """
select dbo.options.*, dbo.equities.Mid as 'Spot', dbo.rates.Rate from dbo.options
join dbo.equities on dbo.options.Date = dbo.equities.Date and dbo.options.Time = dbo.equities.Time 
and dbo.options.Date = '27/05/2020' and dbo.options.Time = '15:20' and dbo.options.Lotsize = 100
join dbo.rates on dbo.options.Date = dbo.rates.Date"""
df = get_frame("Historical",query)
df = trim_grid(df)
df = get_implied_div(df,True,True)
#--------------SVI----------------------------------------------------------------------------
start = time.time()
chi = SVI(df).run()
end = time.time()
print(f"Runtime: {end-start}")
