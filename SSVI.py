from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import log,sqrt,exp,inf,pi
from scipy.interpolate import interp1d
from scipy.stats import norm
import pandas as pd
import numpy as np
import sqlalchemy
cdf,pdf = norm.cdf,norm.pdf
pd.options.mode.chained_assignment = None
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')

def graphs(df,params):
    for t in sorted(df["Tenor"].unique()):
        k = df[df["Tenor"]==t]["LogStrike"]
        v = df[df["Tenor"]==t]["AtmVar"]
        w = df[df["Tenor"]==t]["ImpliedVar"]*(t/365)
        plt.plot(k,surface(k,v,*params),label=t)
        plt.scatter(k,w,s=2,label=t)
    plt.legend(); plt.show()
    K = get_strike_vector(10,0.1,60)
    for t in sorted(df["Tenor"].unique()):
        S = df[df["Tenor"]==t]["Spot"].iloc[0]
        r = df[df["Tenor"]==t]["Rate"].iloc[0]
        q = df[df["Tenor"]==t]["ImpliedDiv"].iloc[0]
        v = df[df["Tenor"]==t]["AtmVar"].iloc[0]
        k = get_logstrike(K,t,S,r,q)
        w = surface(k,v,*params)/(t/365)
        Q = rnd(K,t,S,r,q,w)
        plt.plot(k[1:-1],Q,label=t)
    plt.legend(); plt.show()  
    dfx = df[df["Type"]=="Call"]
    for t in sorted(dfx["Tenor"].unique()):
        v = dfx[dfx["Tenor"]==t]["AtmVar"].iloc[0]
        K = dfx[dfx["Tenor"]==t]["Strike"]
        S = dfx[dfx["Tenor"]==t]["Spot"].iloc[0]
        r = dfx[dfx["Tenor"]==t]["Rate"].iloc[0]
        q = dfx[dfx["Tenor"]==t]["ImpliedDiv"].iloc[0]
        R = dfx[dfx["Tenor"]==t]["Type"].to_numpy()
        k = get_logstrike(K,t,S,r,q)
        ATM = np.vectorize(get_value)("Call",S,S,t,r,q,sqrt(v/(t/365)))
        w = surface(k,v,*params)/(t/365)
        P = np.vectorize(get_value)(R,S,K,t,r,q,sqrt(w))
        plt.plot(K,P,label=t)
        payoff = [max(0,S-k) for k in K]
        plt.plot(K,payoff,label="Payoff",c="k",linewidth=1)
        plt.scatter(K,dfx[dfx["Tenor"]==t]["Mid"],label=t,s=2)
        plt.scatter(S,ATM,c="k",s=4)
    plt.legend(); plt.show()
    dfx = df[df["Type"]=="Put"]
    for t in sorted(dfx["Tenor"].unique()):
        v = dfx[dfx["Tenor"]==t]["AtmVar"].iloc[0]
        K = dfx[dfx["Tenor"]==t]["Strike"]
        S = dfx[dfx["Tenor"]==t]["Spot"].iloc[0]
        r = dfx[dfx["Tenor"]==t]["Rate"].iloc[0]
        q = dfx[dfx["Tenor"]==t]["ImpliedDiv"].iloc[0]
        R = dfx[dfx["Tenor"]==t]["Type"].to_numpy()
        k = get_logstrike(K,t,S,r,q)
        ATM = np.vectorize(get_value)("Put",S,S,t,r,q,sqrt(v/(t/365)))
        w = surface(k,v,*params)/(t/365)
        P = np.vectorize(get_value)(R,S,K,t,r,q,sqrt(w))
        plt.plot(K,P,label=t)
        payoff = [max(0,k-S) for k in K]
        plt.plot(K,payoff,label="Payoff",c="k",linewidth=1)
        plt.scatter(K,dfx[dfx["Tenor"]==t]["Mid"],label=t,s=2)
        plt.scatter(S,ATM,c="k",s=4)
    plt.legend(); plt.show()
    #Errors
    call_errors = pd.DataFrame(columns=["t","c_error"])
    dfx = df[df["Type"]=="Call"]
    for t in sorted(dfx["Tenor"].unique()):
        v = dfx[dfx["Tenor"]==t]["AtmVar"].iloc[0]
        K = dfx[dfx["Tenor"]==t]["Strike"]
        S = dfx[dfx["Tenor"]==t]["Spot"].iloc[0]
        r = dfx[dfx["Tenor"]==t]["Rate"].iloc[0]
        q = dfx[dfx["Tenor"]==t]["ImpliedDiv"].iloc[0]
        R = dfx[dfx["Tenor"]==t]["Type"].to_numpy()
        k = get_logstrike(K,t,S,r,q)
        ATM = np.vectorize(get_value)("Call",S,S,t,r,q,sqrt(v/(t/365)))
        w = surface(k,v,*params)/(t/365)
        P = np.vectorize(get_value)(R,S,K,t,r,q,sqrt(w))
        error = abs(dfx[dfx["Tenor"]==t]["Mid"]-P)
        spread = dfx[dfx["Tenor"]==t]["Spread"]/2
        spread = [max(0.,e-s) for e,s in zip(error,spread)]
        plt.plot(K,spread,label=t)
        call_errors.loc[len(call_errors)] = [t,sum(spread)/len(spread)]
    plt.legend(); plt.show()
    put_errors = pd.DataFrame(columns=["t","p_error"])
    dfx = df[df["Type"]=="Put"]
    for t in sorted(dfx["Tenor"].unique()):
        v = dfx[dfx["Tenor"]==t]["AtmVar"].iloc[0]
        K = dfx[dfx["Tenor"]==t]["Strike"]
        S = dfx[dfx["Tenor"]==t]["Spot"].iloc[0]
        r = dfx[dfx["Tenor"]==t]["Rate"].iloc[0]
        q = dfx[dfx["Tenor"]==t]["ImpliedDiv"].iloc[0]
        R = dfx[dfx["Tenor"]==t]["Type"].to_numpy()
        k = get_logstrike(K,t,S,r,q)
        ATM = np.vectorize(get_value)("Put",S,S,t,r,q,sqrt(v/(t/365)))
        w = surface(k,v,*params)/(t/365)
        P = np.vectorize(get_value)(R,S,K,t,r,q,sqrt(w))
        error = abs(dfx[dfx["Tenor"]==t]["Mid"]-P)
        spread = dfx[dfx["Tenor"]==t]["Spread"]/2
        spread = [max(0.,e-s) for e,s in zip(error,spread)]
        plt.plot(K,spread,label=t)
        put_errors.loc[len(put_errors)] = [t,sum(spread)/len(spread)]        
    plt.legend(); plt.show()
    return call_errors.set_index("t").join(put_errors.set_index("t"))
    
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

def get_value(R,S,X,T,r,q,s): 
    T = T/365
    d1 = (log(S/X)+(r-q+(s**2)/2)*T)/(s*sqrt(T))
    d2 = d1-(s*sqrt(T))
    if R == "Call":
        return (S*exp(-q*T)*cdf(d1))-(X*exp(-r*T)*cdf(d2))
    else:
        return (X*exp(-r*T)*cdf(-d2))-(S*exp(-q*T)*cdf(-d1))
        
def get_frame(database,query):
    server="LAPTOP-206OR7PL\\SQLEXPRESS"
    engine = sqlalchemy.create_engine('mssql+pyodbc://@'+server+'/'+
    database+'?trusted_connection=yes&driver=ODBC+Driver+13+for+SQL+Server')
    return pd.read_sql(query,con=engine)

def get_tenor(df,cols):
    for x in cols: df[x] = pd.to_datetime(df[x],format="%d/%m/%Y")
    df["Tenor"] = (df[cols[1]]-df[cols[0]]).dt.days
    for x in cols: df[x] = df[x].apply(lambda x:x.strftime("%d/%m/%Y"))
    return df

def get_forward(T,S,r,q): return S*exp((-r-q)*(T/365))
    
def get_logstrike(K,T,S,r,q): return log(K/(S*exp((-r-q)*(T/365))))

def get_strike_vector(minK,dK,maxK): return np.array([i*dK for i in range(int(minK/dK),int(maxK/dK)+3)])

def raw(x,a,b,rho,m,sigma): return a+b*(rho*(x-m)+sqrt((x-m)**2+sigma**2))

def correlation(theta,a,b,c): return a*exp(-b*theta)+c

def power_law(theta,eta,gamma): 
    return eta/theta**gamma*(1+theta)**(1-gamma)

def fly_constraint1(params,theta):
    a,b,c,eta,gamma = params
    rho = correlation(theta,a,b,c)
    phi = power_law(theta,eta,gamma)
    return (theta*phi*(1+abs(rho)))+4

def fly_constraint2(params,theta): 
    a,b,c,eta,gamma = params
    rho = correlation(theta,a,b,c)
    phi = power_law(theta,eta,gamma)
    return (theta*phi**2*(1+abs(rho)))+(4+1e-10)
    
def eta_constraint3(params,theta):
    a,b,c,eta,gamma = params
    rho = correlation(theta,a,b,c)
    return eta*(1-abs(rho))+(2+1e-10)

def eta_constraint1(params,theta):
    a,b,c,eta,gamma = params
    rho = correlation(theta,a,b,c)
    return eta+(4*max(theta)**(gamma-1))/(1+abs(rho))

def eta_constraint2(params,theta):
    a,b,c,eta,gamma = params
    rho = correlation(theta,a,b,c)
    return eta+(2*max(theta)**(gamma-1/2))/(sqrt(1+abs(rho)))
    
def rho_constraint(params,theta):
    a,b,c,eta,gamma = params
    rho = correlation(theta,a,b,c)
    return abs(rho)+1

def gamma_constraint(params,theta):
    a,b,c,eta,gamma = params
    rho = correlation(theta,a,b,c)
    return gamma+((1+sqrt(1-rho**2))/rho**2)

def abc_constraint(params):
    a,b,c,eta,gamma = params
    return abs((c-a)/(1-gamma*exp(gamma-2)))+1

def ac_constraint(params):
    a,b,c,eta,gamma = params
    return abs(a+c)+1

def surface(k,theta,a,b,c,eta,gamma):
    rho = correlation(theta,a,b,c)
    phi = power_law(theta,eta,gamma)
    return (theta/2)*(1+rho*phi*k+sqrt((k*phi+rho)**2+(1-rho**2)))

#Breeden&Litzenberger density: taking second derivative of call price to strike
def rnd(K,T,S,r,q,w):
    dK = (K[1]-K[0])
    C = get_value("Call",S,K,T,r,q,sqrt(w))
    Q = [exp(r*(T/365))*(c2-(2*c1)+c0)/dK for c2,c1,c0 in zip(C[2:],C[1:-1],C[:-2])]
    return Q

#Minimise weighted absolute distance from mid prices
def residual(params,df,k,v,t):
    S = df["Spot"].iloc[0]
    mid_raw,vol = df["Mid"].to_numpy(),sqrt(surface(k,v,*params)/(t/365))
    mid_svi = np.vectorize(get_value)(df["Type"],df["Spot"],df["Strike"],df["Tenor"],df["Rate"],df["ImpliedDiv"],vol)
    weights = [1-((i-S)/S)**2 for i in df["Strike"]] #Squared distance from the spot
    epsilon = sum([abs(r-s)*w for r,s,w in zip(mid_raw,mid_svi,weights)])**2
    if np.isnan(epsilon) == True:
        return 1e3
    else:
        return epsilon

def optimise_surface(df):
    k = df["LogStrike"].to_numpy()
    v = df["AtmVar"].to_numpy()
    t = df["Tenor"].to_numpy()
    guess = (-0.2,10,-0.2,0.5,0.25)
    bounds = ((-inf,inf),(0,inf),(-1,1),(0,1),(0,0.5))
    res = minimize(residual,guess,args=(df,k,v,t),method="SLSQP",options={"maxiter":5000},bounds=bounds,
    constraints=({"type":"ineq","fun":eta_constraint1,"args":(v,)},
                 {"type":"ineq","fun":eta_constraint2,"args":(v,)},
                 {"type":"ineq","fun":eta_constraint3,"args":(v,)},
                 {"type":"ineq","fun":fly_constraint1,"args":(v,)},
                 {"type":"ineq","fun":fly_constraint2,"args":(v,)},
                 {"type":"ineq","fun":rho_constraint,"args":(v,)},
                 {"type":"ineq","fun":gamma_constraint,"args":(v,)},
                 {"type":"ineq","fun":abc_constraint},
                 {"type":"ineq","fun":ac_constraint}))
    chi = pd.DataFrame(columns=["S","t","r","q","theta","a","b","c","eta","gamma"])
    S,r = df["Spot"].iloc[0],df["Rate"].iloc[0]
    for t,q,v in zip(df["Tenor"].unique(),df["ImpliedDiv"].unique(),df["AtmVar"].unique()):
        chi.loc[len(chi)] = [S,t,r,q,v,*res.x]
    return chi.set_index("t").join(graphs(df,res.x)).reset_index()

def trim_grid(df):
    df = get_tenor(df,["Date","Expiry"])
    cols = ["Strike","Bid","Ask","Spread","Mid"]
    df[cols] = df[cols].astype(float)
    df["Moneyness"] = df["Strike"]/df["Spot"]
    df = df[(df["Tenor"]!=0)&(df["Bid"]>0.1)&(df["Ask"]>0.1)] #&(df["Moneyness"]>0.7)&(df["Moneyness"]<1.3)]
    return df
  
def get_atm(df,plot=False):
    df["LogStrike"] = np.vectorize(get_logstrike)(df["Strike"],df["Tenor"],df["Spot"],df["Rate"],df["ImpliedDiv"])
    df["ImpliedVar"] = np.vectorize(bisect)(df["Type"],df["Mid"],df["Spot"],df["Strike"],df["Tenor"],df["Rate"],df["ImpliedDiv"])**2
    df = df.dropna()
    dfw = pd.DataFrame(columns=["Tenor","AtmVar"])
    for t in sorted(df["Tenor"].unique()):
        dfx = df[df["Tenor"]==t].drop_duplicates(["LogStrike"]).sort_values(by=["LogStrike"])
        k = dfx["LogStrike"].to_numpy()
        w = dfx["ImpliedVar"].to_numpy()
        interp = interp1d(k,w,kind="cubic")
        theta = interp(0)*(t/365)
        dfw.loc[len(dfw)] = [t,theta]
    df = df.set_index("Tenor").join(dfw.set_index("Tenor")).reset_index()
    if plot==True: 
        print(dfw)
        plt.plot(dfw["Tenor"],dfw["AtmVar"])
        plt.show()
    return df.sort_values(by=["Strike"])

def get_implied_div(df,ex_itm_puts=False,plot=False):
    dfx = df[df["Moneyness"]>=1] if ex_itm_puts==True else df
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

query = """
select dbo.options.*, dbo.equities.Mid as 'Spot', dbo.rates.Rate from dbo.options
join dbo.equities on dbo.options.Date = dbo.equities.Date and dbo.options.Time = dbo.equities.Time 
and dbo.options.Date = '06/05/2020' and dbo.options.Time = '15:20' and dbo.options.Lotsize = 10
join dbo.rates on dbo.options.Date = dbo.rates.Date"""
df = get_frame("Historical",query)
df = trim_grid(df)
df = get_implied_div(df,True,True).dropna()
df = get_atm(df,True)
params = optimise_surface(df)
