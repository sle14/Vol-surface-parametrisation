from numpy.lib.recfunctions import unstructured_to_structured as struct
from numpy.lib.recfunctions import structured_to_unstructured as unstruct
from numpy import log,sqrt,exp,inf,nan,pi
import plotly.graph_objects as go
from ipywidgets import widgets
import traitlets as tl
import pandas as pd
import numpy as np
import calibrator
import pricer
import query
import utils

cout = utils.log(__file__,__name__,disp=False)
base = 253

class theme:
    def __init__(self):
        self.gray = "rgb(40,40,40)"
        self.black = "rgb(10,10,10)"
        self.orange = "rgb(255,127,80)"
        self.darkgray = "rgb(30,30,30)"
        self.lightgray = "rgb(50,50,50)"
        self.white = "rgb(255,255,255)"
        self.darkwhite = "rgb(120,120,120)"
        self.template = "plotly_dark"
        self.transparent = "rgba(0,0,0,0)"
        data_style = f"""
        <style>.p-TabBar-tab 
                |# color:{self.white} !important; background-color:{self.darkgray} !important; border-color:{self.lightgray} !important; #|
        </style>
        <style>.widget-tab-contents
                |# color:{self.white} !important; background-color:{self.darkgray} !important; border-color:{self.lightgray} !important; #|
        </style>
        <style>.output_area pre
                |# color:{self.orange} !important; #|
        </style>        
        <style>.widget-text input[type="text"] 
                |# color:{self.white}; background-color:{self.gray}; border-color:{self.gray}; #|
        </style>
        <style>.widget-text input[type="number"]
                |# color:{self.white}; background-color:{self.gray}; border-color:{self.gray}; #|
        </style>
        <style>.widget-dropdown > select 
                |# color:{self.white}; background-color:{self.gray}; border-color:{self.gray}; #|
        </style>
        <style>.widget-toggle-button 
                |# color:{self.white}; background-color:{self.gray}; border-color:{self.gray}; #|
        </style>
        <style>.jupyter-button.mod-active
                |# color:{self.darkwhite}; background-color:{self.gray}; border-color:{self.gray}; #|
        </style>
        <style>.widget-button 
                |# color:{self.white}; background-color:{self.gray}; border-color:{self.gray}; #|
        </style>
        <style>.widget-datepicker input[type="date"] 
                |# color:{self.white}; background-color:{self.gray}; border-color:{self.gray}; #|
        </style>
        <style>.progress-bar
                |# background-color:{self.orange} !important; #|
        </style>
        <style>.progress
                |# background-color:{self.darkgray} !important; #|
        </style>
        """
        self.data_style = data_style.replace("|#","{").replace("#|","}")

c = theme()
#------------------------------------------------------------------------------
vol_cols = [
            "Tenor",
            "Strike",
            "LogStrike",
            "Moneyness",
            "RawVol",
            "SmtVol",
            "CallBid",
            "CallAsk",
            "CallMid",
            "CallSpread",
            "CallEEP",
            "PutBid",
            "PutAsk",
            "PutMid",
            "PutSpread",
            "PutEEP"
           ]
fct_cols = [       #Independent vars:
            "TNR", #Tenor
            "CDD", #Cum-div days
            "SPT", #Spot px
            "ATM", #ATMF Vol
            "SKW", #ATMF Skew
            "KRT", #ATMF Kurt
            "DIV", #Cash div
            "BOR", #Imp borrow
            "FRT", #Funding rate
            "CH0", #Call spread min
            "CH1", #Call spread slope
            "CKH", #Call spread strike
            "PH0", #Put spread min
            "PH1", #Put spread slope
            "PKH", #Put spread strike
            "PHI", #Convexity = phi
            "RHO", #Skew term = rho
           ]
out_cols = [       #Dependent vars:
            "LST", #Log-strike
            "MON", #Moneyness
            "FWD", #Forward px
            "YLD", #Implied yield
            "VOL", #Implied vol
            "VAR", #Total variance
            "RND", #Risk neutral density
            "CPX", #Call px
            "PPX", #Put px
            "CEP", #Call early exercise premium
            "PEP", #Put early exercise premium
            "CHS", #Call half spread
            "PHS", #Put half spread
            "CDL", #Call delta
            "PDL", #Put delta
            "GAM", #Gamma
            "VEG", #Vega
            "CTH", #Call theta
            "PTH", #Put theta
            "CRH", #Call rho
            "PRH", #Put rho
            "SPD", #Speed
            "VOM", #Vomma
            "VAN", #Vanna
            "CCH", #Call charm
            "PCH", #Put charm
            "CPY", #Call payoff
            "PPY", #Put payoff
            "CER", #Call px errors, fitted vs raw
            "PER", #Put px errors, fitted vs raw
            "VER", #Vall errors, fitted vs raw
            "EED", #Early exercise deviation
           ]
raw_cols = [        #Raw vars
            "RVOL", #Vol
            "RCMD", #Call mid
            "RCHS", #Call half bid-ask spread
            "RPMD", #Put mid
            "RPHS", #Put half bid-ask spread
            "RCEP", #Call ame-eur spread (raw px - fitted px)
            "RPEP", #Put ame-eur spread (raw px - fitted px)
           ]    
pay_cols = [        #Return vars
            "PCTM", #Percent dS
            "LQPX", #Liquidation px
            "LQEP", #Liquidation eep
            "LQHS", #Liquidation spread costs
            "EXPX", #Exercise px
            "EXEP", #Exercise eep
            "EXHS", #Exercise spread costs
            "SPTR"  #Liquidation spot
           ]
trm_cols = [        #Term vars
            "DPAS", #Days passed
            "LQPX", #Liquidation px
            "LQEP", #Liquidation eep
            "LQHS", #Liquidation spread costs
            "EXPX", #Exercise px
            "EXEP", #Exercise eep
            "EXHS", #Exercise spread costs
            "SPTR"  #Liquidation spot
           ]
trd_cols = [       #Trade vars
            "RGT", #Right: 1=Call, 0=Put, -1=Spot
            "LEG", #Leg: 0=Open, 1=Close
            "IDX", #Expiry group: 0,1,2,3...
            "STR", #Strike: 0 for spot
            "SIZ", #Size of position/notional
           ]

greeks_headers = ["CDL","PDL","GAM","VEG","CTH","PTH","CRH","PRH","SPD","VOM","VAN","CCH","PCH"]
greeks_attrib = ["delta","delta","gamma","vega","theta","theta","rho","rho","speed","vomma","vanna","charm","charm"]
greeks_rights = ["C","P","x","x","C","P","C","P","x","x","x","C","P"]

def symbol_universe():
    return query.get("Static","select distinct Symbol from dbo.chains where Symbol != 'FP' order by Symbol")["Symbol"].tolist()

def trading_times(symbol):
    tms = query.get("Vols",f"select distinct Time from dbo.{symbol} where Date = convert(datetime,'26/10/2020',103)")["Time"].tolist()
    return [str(x)[:2]+":"+str(x)[2:]+" UTC" for x in list(sorted(tms))]

class Progress(tl.HasTraits):
    val = tl.Float

prg_load = Progress()

class ChainStruct:
    def __init__(self,params,vols,k):
        self.prg_pay = Progress()                                               #Payoff build exec progress
        self.prg_trm = Progress()                                               #Term build exec progress
        
        self.days_pass = np.arange(0,max(vols.index)+1,1)                       #Days deducted from for dTenor
        self.pct_moves = np.linspace(-0.5,0.5,101)                              #Pct change vector for dSpot
        self.params = params                                                    #Params df with fitted vol params
        self.vols = vols                                                        #Vols df with raw vals per strike
        self.groups = len(params)                                               #Expiry group count
        self.k = k                                                              #Logstrike vector
        
        self.clear_factor()                                                     #Factor struct
        self.ss = self.pack_smt()                                               #Smooth k struct
        self.rs = self.pack_raw()                                               #Raw struct
        self.ps = self.pack_pay()                                               #Payoff/return struct
        self.ns = self.pack_trm()                                               #Term struct
        self.ts = self.pack_trd()                                               #Trades struct
        
        prg_load.val += 0.1
        self.pop(0)                                                             #Pop open leg
        prg_load.val += 0.1
        self.pop(1)                                                             #Pop close leg
        prg_load.val += 0.1
        
    def pack_pay(self):
        cols = ([(x,float) for x in pay_cols])
        ps = np.zeros(self.pct_moves.shape[0],dtype=cols)
        ps["PCTM"] = self.pct_moves
        return ps

    def pack_trd(self):
        header = [(x,y) for x,y in zip(trd_cols,[int,int,int,float,int])]
        ts = np.zeros(100,dtype=header)
        return ts

    def pack_trm(self):
        cols = ([(x,float) for x in trm_cols])
        ps = np.zeros(self.days_pass.shape[0],dtype=cols)
        ps["DPAS"] = self.days_pass
        return ps
        
    def pack_smt(self):
        header = ([(x,np.float64,(2,)) for x in ["LST","STR","MON","VAR","RND","VOL"]]+
                  [(x,np.float64,(2,)) for x in greeks_headers])
        st = np.zeros(len(self.k),dtype=header)
        st["LST"][:,0] = self.k
        st["LST"][:,1] = self.k
        st = np.repeat(st[:,np.newaxis],self.groups,axis=1)
        return st
        
    def pack_raw(self):
        header = ([("IDX",np.int32)]+[("STR",np.float64)]+[(x,np.float64) for x in fct_cols]+
                  [(x,np.float64) for x in out_cols]+[(x,np.float64) for x in raw_cols])
        st = np.zeros(len(self.vols),dtype=header)
        st = np.repeat(st[:,np.newaxis],2,axis=1)
        st = self.index(st)
        vols_cols = ["RawVol","CallMid","CallSpread","PutMid","PutSpread"]
        for x,y in zip(raw_cols[:-2],vols_cols): st[x][:,0] = self.vols[y]
        st["RCHS"][:,0] = st["RCHS"][:,0]/2
        st["RPHS"][:,0] = st["RPHS"][:,0]/2
        return st

    def index(self,st):                                                         #Transform tenor value to expiry group num
        idx = np.array(list(sorted(self.params["TNR"])))
        idx = np.repeat(idx[:,np.newaxis],2,axis=1)
        idx[:,1] = range(idx.shape[0])
        dct = dict(zip(idx[:,0],idx[:,1]))
        idx = [dct[x] for x in self.vols.index]
        st["STR"][:,0] = self.vols["Strike"]
        st["STR"][:,1] = self.vols["Strike"]
        st["IDX"][:,0] = idx
        st["IDX"][:,1] = idx
        return st

    def search_raw(self,I,K,cols,leg=0):                                        #Arrays have to be sorted for this!
        st = self.rs[cols][np.where(
                           np.logical_and(
                           np.in1d(self.rs["IDX"][:,leg],I),
                           np.in1d(self.rs["STR"][:,leg],K)
                           ))[0],leg]
        return unstruct(st)
    
    def groupby(self,st,idx,cols,nozero_col):                                   #To be changed to full numpy
        df = pd.DataFrame(st[idx]).groupby(cols).sum()
        dt = df.reset_index().values
        dt = dt[np.where(dt[:,nozero_col]!=0)[0],:]
        st[list(st.dtype.names)] = 0
        st[list(st.dtype.names)][0:dt.shape[0]] = [tuple(x) for x in dt]
        return st
    
    def append_trd(self,RGT,LEG,IDX,STR,SIZ):                                   #Add trade to beginning of zeros part of struct
        ts = self.ts
        ts[np.where(ts["SIZ"]==0)[0][0]] = (RGT,LEG,IDX,STR,SIZ)
        self.ts = self.groupby(ts,np.where(ts["SIZ"]!=0)[0],["RGT","LEG","IDX","STR"],-1)

    def liquidate_trd(self):                                                    #Create offsets for positions that are not closed out, (does not care about leg size atm)
        st = self.ts[np.where(self.ts["SIZ"]!=0)[0]]
        arr,idx,cnt = np.unique(st[["RGT","IDX","STR"]],return_index=True,return_counts=True)
        new = st[idx[np.where(cnt==1)[0]]]
        new["LEG"] = np.where(new["LEG"]==1,0,1)
        new["SIZ"] = new["SIZ"] * -1
        idx = np.where(self.ts["SIZ"]==0)[0]
        self.ts[idx[0:new.shape[0]]] = new
        self.ts[idx] = np.sort(self.ts[idx],order=["LEG","RGT","IDX"])

    def join_search(self,cols):                                                 #Join trades with cols requested
        ts = self.ts
        ts = unstruct(ts[np.where(ts["SIZ"]!=0)[0]]) 
        stk = ts[np.where(ts[:,0]==-1)[0],:]
        opt = ts[np.where(ts[:,0]!=-1)[0],:]
        opt = utils.sort_array(opt,(3,2,1,0))
        lst = []
        cols = ["IDX","STR","TNR"] + cols
        for l in range(2):
            I = opt[np.where(opt[:,1]==l)[0],2]
            K = opt[np.where(opt[:,1]==l)[0],3]
            st = self.search_raw(I,K,cols,l)
            leg = np.zeros((st.shape[0],1))
            leg[:] = l 
            st = np.hstack([leg,st])                                            #join leg
            lst.append(st)
        st = np.vstack(lst) 
        st = np.vstack([                                                        #join right
             np.hstack([np.ones((st.shape[0],1)),st]),                          #Calls
             np.hstack([np.zeros((st.shape[0],1)),st])                          #Puts
                       ])
        st = st[utils.intersect2d(st[:,(0,1,2,3)],opt[:,(0,1,2,3)]),:]
        st = utils.sort_array(st,(3,2,1,0)) 
        size = np.expand_dims(opt[:,-1],axis=0).T
        opt = np.hstack([size,st])                                              #join size
        return opt,stk                                                          #STK: RGT,LEG,IDX,STR,SIZ & OPT: SIZ,RGT,LEG,IDX,STR,TNR + cols requested

    def unpack_trd(self,st):
        rights = {1:"Call",0:"Put",-1:"Spot"}
        out = [x.tolist() for x in unstruct(st).T]
        out[0] = ["Open" if x==0 else "Close" for x in out[0]]                  #Leg
        out[2] = ["Ask" if x==1 else "Bid" for x in out[2]]                     #Side
        out[3] = [rights[x] for x in out[3]]                                    #Right
        return out

    def build_trd(self):
        cols = ["CPX","PPX","CDL","PDL","GAM","VEG","CTH","PTH","VOL","CHS",
                "PHS","CEP","PEP","VOM","VAN"]
        opt,stk = self.join_search(cols)

        opt_LEG = opt[:,2]
        opt_TNR = opt[:,5].astype(int)
        opt_SID = np.where(opt[:,0]<0,-1,1)
        opt_RGT = opt[:,1]
        opt_SIZ = opt[:,0].astype(int)
        opt_STR = opt[:,4]
        opt_PPX = np.where(opt_RGT==1,opt[:,6],opt[:,7]) * -opt_SIZ
        opt_DEL = np.where(opt_RGT==1,opt[:,8],opt[:,9]) * opt_SIZ
        opt_GAM = opt[:,10] * opt_SIZ
        opt_VEG = opt[:,11] * opt_SIZ
        opt_THT = np.where(opt_RGT==1,opt[:,12],opt[:,13]) * opt_SIZ
        opt_VOL = opt[:,14]
        opt_SPR = np.where(opt_RGT==1,opt[:,15],opt[:,16]) * -abs(opt_SIZ)
        opt_EEP = np.where(opt_RGT==1,opt[:,17],opt[:,18]) * -opt_SIZ
        opt_VOM = opt[:,19] * opt_SIZ
        opt_VAN = opt[:,20] * opt_SIZ
        opt_out = np.array([opt_LEG,opt_TNR,opt_SID,opt_RGT,opt_SIZ,opt_STR,opt_PPX,opt_DEL,
                            opt_GAM,opt_VEG,opt_THT,opt_VOM,opt_VAN,opt_VOL,opt_SPR,opt_EEP])

        stk_LEG = stk[:,1]
        stk_TNR = np.zeros(stk.shape[0]).astype(int)
        stk_SID = np.where(stk[:,4]<0,-1,1)
        stk_RGT = stk[:,0]
        stk_SIZ = stk[:,4].astype(int)
        stk_nan = np.zeros(stk.shape[0])
        stk_PPX = np.where(stk_LEG==1,self.get_raw(0,"SPT",1)[0],
                           self.get_raw(0,"SPT",0)[0]) * -stk_SIZ
        stk_DEL = stk[:,4].astype(float)
        stk_out = np.array([stk_LEG,stk_TNR,stk_SID,stk_RGT,stk_SIZ,stk_nan,stk_PPX,stk_DEL,
                            stk_nan,stk_nan,stk_nan,stk_nan,stk_nan,stk_nan,stk_nan,stk_nan])

        out = np.vstack([opt_out.T,stk_out.T])
        out = utils.sort_array(out,(1,3,0))
        
        head = ["LEG","TNR","SID","RGT","SIZ","STR","PPX","DEL","GAM","VEG","THT","VOM","VAN","VOL","SPR","EEP"]
        dtyp = [int  ,int  ,int  ,int  ,int  ,float,float,float,float,float,float,float,float,float,float,float]
        dtype = np.dtype([(x,y) for x,y in zip(head,dtyp)])
        return struct(out,dtype=dtype)

    def totals(self,st):
        st = np.asarray(np.split(st,2)).T                                       #We need only open data
        
        dS = self.get_raw(0,"SPT",1)[0] - self.get_raw(0,"SPT",0)[0]
        dT = st["TNR"][:,1] - st["TNR"][:,0]
        ds = st["VOL"][:,1] - st["VOL"][:,0]

        pnl_SIZ = sum(st["SIZ"][:,1] + st["SIZ"][:,0])
        pnl_PPX = sum(st["PPX"][:,1] + st["PPX"][:,0])
        pnl_DEL = sum([x*dS for x in st["DEL"][:,0]])
        pnl_GAM = sum([x*0.5*dS**2 for x in st["GAM"][:,0]])
        pnl_VEG = sum([x*ds_*100 for x,ds_ in zip(st["VEG"][:,0],ds)])
        pnl_THT = sum([x*dt for x,dt in zip(st["THT"][:,0],dT)])
        pnl_SPR = sum(st["SPR"][:,1] + st["SPR"][:,0])
        pnl_EEP = sum(st["EEP"][:,1] + st["EEP"][:,0])
        pnl_VOM = sum([0.5*x*ds_**2*100 for x,ds_ in zip(st["VOM"][:,0],ds)])
        pnl_VAN = sum([x*dS*ds_ for x,ds_ in zip(st["VAN"][:,0],ds)])

        exp_SIZ = sum(st["SIZ"][:,0])
        exp_PPX = sum(st["PPX"][:,0])
        exp_DEL = sum(st["DEL"][:,0])
        exp_GAM = sum(st["GAM"][:,0])
        exp_VEG = sum(st["VEG"][:,0])
        exp_THT = sum(st["THT"][:,0])
        exp_VOM = sum(st["VOM"][:,0])
        exp_VAN = sum(st["VAN"][:,0])
        exp_SPR = sum(st["SPR"][:,0])
        exp_EEP = sum(st["EEP"][:,0])

        exp_row = ["","","","OpenExp:",exp_SIZ,"",exp_PPX,exp_DEL,exp_GAM,exp_VEG,exp_THT,exp_VOM,exp_VAN,"",exp_SPR,exp_EEP]    
        pnl_row = ["","","","PnL:",pnl_SIZ,"",pnl_PPX,pnl_DEL,pnl_GAM,pnl_VEG,pnl_THT,pnl_VOM,pnl_VAN,"",pnl_SPR,pnl_EEP]
        pnl_clr = [c.white if type(x) is str else "red" if x<0 else "green" if x>0 else c.white for x in pnl_row]
        return exp_row,pnl_row,pnl_clr

    def get_raw(self,i,col,leg=0,K=None):
        if K is None:
            if col != "STR":
                return self.rs[col][np.where(self.rs["IDX"][:,0]==i),leg].ravel()
            else:
                return self.rs[col][np.where(self.rs["IDX"][:,0]==i),0].ravel()
        else:
            return self.rs[col][np.where((self.rs["IDX"][:,0]==i)&
                   (np.in1d(self.rs["STR"][:,0],K))),leg].ravel()

    def get_smt(self,i,col,leg=0,atmf=False):
        if atmf == False:
            if col != "LST":
                return self.ss[col][:,i,leg].ravel()
            else:
                return self.k
        else:
            if col != "LST":
                return self.ss[col][np.where(self.k==0)[0],i,leg].ravel()
            else:
                return [0]                                                      #Logstrike at atmf == 0
        
    def clear_factor(self):
        header = ([("TNR",np.int32),("CDD",np.int32)]+[(x,np.float64) for x in fct_cols[2:]])
        self.fs = np.zeros(self.groups,dtype=header)
        
    def pop(self,leg=0):
        rs,fs = self.rs,self.fs
        if leg==0: 
            rs[fct_cols][:,0] = self.params.loc[rs["IDX"][:,0],fct_cols].to_records(index=False)
            rs[fct_cols][:,1] = self.params.loc[rs["IDX"][:,0],fct_cols].to_records(index=False)
            self.rs = rs
            self.build_raw(0)
        else:
            rs[fct_cols[2:]][:,1] = struct(                                     #Have to convert to unstructured array for broadcasting across cols of the structure
                                           (unstruct(rs[fct_cols[2:]][:,0]) * 
                                           (1 + unstruct(fs[fct_cols[2:]][rs["IDX"][:,0]]))),
                                           dtype = (rs[fct_cols[2:]][:,1].dtype)
                                          )
            rs[fct_cols[:2]][:,1] = struct(
                                           (unstruct(rs[fct_cols[:2]][:,0]) - 
                                           unstruct(fs[fct_cols[:2]][rs["IDX"][:,0]])),
                                           dtype = (rs[fct_cols[:2]][:,1].dtype)
                                           )     
            self.rs = rs
            self.build_raw(1)
            
    def ki_args(self,st,R="C",leg=0):                                           #KimIntegral args helper
        args = [st["SPT"][:,leg],
                st["STR"][:,0],
                st["TNR"][:,leg]/base,
                st["VOL"][:,leg],
                st["FRT"][:,leg],
                st["YLD"][:,leg],
                st[f"{R}H0"][:,leg],
                st[f"{R}H1"][:,leg],
                st[f"{R}KH"][:,leg],]
        return args
    
    def build_smt(self,params,leg=0):
        ss,rs,k,sv = self.ss,self.rs,self.k,pricer.SSVI()
        F = utils.unique_nosort(rs["FWD"][:,leg])
        T = utils.unique_nosort(rs["TNR"][:,leg]/base)
        q = utils.unique_nosort(rs["YLD"][:,leg])
        tht = utils.unique_nosort(params[0])
        phi = utils.unique_nosort(params[1])
        rho = utils.unique_nosort(params[2])
        K = np.vstack([F[i]*exp(k) for i in range(self.groups)])
        S = rs["SPT"][:,leg][0]
        r = rs["FRT"][:,leg][0]
        for i in range(self.groups):
            ss["STR"][:,i,leg] = K[i]
            ss["MON"][:,i,leg] = K[i]/S
            if T[i] <= 0:
                ss["VAR"][:,i,leg] = 0
                ss["RND"][:,i,leg] = 0
                ss["VOL"][:,i,leg] = 0
                for g,a,R in zip(greeks_headers,greeks_attrib,greeks_rights):
                    ss[g][:,i,leg] = 0
            else:
                ss["VAR"][:,i,leg] = sv.raw(k,tht[i],phi[i],rho[i])
                ss["RND"][:,i,leg] = sv.rnd(k,tht[i],phi[i],rho[i])
                s = ss["VOL"][:,i,leg] = sqrt(ss["VAR"][:,i,leg]/T[i])
                for g,a,R in zip(greeks_headers,greeks_attrib,greeks_rights):
                    bsm = pricer.BlackScholes(R,r,q[i],s)
                    ss[g][:,i,leg] = getattr(bsm,a)(S,K[i],T[i])
        self.ss = ss
    
    def build_raw(self,leg=0):
        rs,sv = self.rs,pricer.SSVI()
        S = rs["SPT"][:,leg]
        K = rs["STR"][:,0]
        T = rs["TNR"][:,leg]/base
        Td = rs["CDD"][:,leg]/base
        r = rs["FRT"][:,leg]
        b = rs["BOR"][:,leg]
        F = rs["FWD"][:,leg]
        k = rs["LST"][:,leg]
        s = rs["VOL"][:,leg]
        rs["MON"][:,leg] = K/S

        if Td[0] < 0: 
            D = rs["DIV"][:,leg] = 0
            q = rs["YLD"][:,leg] = b
        else:
            D = rs["DIV"][:,leg]
            q = rs["YLD"][:,leg] = pricer.dsc2yld(S,T,D,Td,r) + b

        rs["FWD"][:,leg] = pricer.forward(T,S,r,b,Td,D)
        rs["LST"][:,leg] = log(K/F)
        rs["CPY"][:,leg] = np.maximum(S-K,0)
        rs["PPY"][:,leg] = np.maximum(K-S,0)

        params = sv.jw2raw(T,rs["ATM"][:,leg],rs["SKW"][:,leg],rs["KRT"][:,leg])
        params = np.nan_to_num(params)
        rs["PHI"][:,leg] = params[1]
        rs["RHO"][:,leg] = params[2]
        rs["VAR"][:,leg] = sv.raw(k,*params)
        rs["VOL"][:,leg] = sqrt(rs["VAR"][:,leg]/T)
        rs["RND"][:,leg] = sv.rnd(k,*params)
        self.build_smt(params,leg)

        rs["CPX"][:,leg],rs["CEP"][:,leg] = pricer.ThreadPool("C",*self.ki_args(rs,"C",leg)).run()
        rs["PPX"][:,leg],rs["PEP"][:,leg] = pricer.ThreadPool("P",*self.ki_args(rs,"P",leg)).run()
        if leg==0:
            rs["RCEP"][:,0] = rs["RCMD"][:,0] - rs["CPX"][:,0]
            rs["RPEP"][:,0] = rs["RPMD"][:,0] - rs["PPX"][:,0]

        for g,a,R in zip(greeks_headers,greeks_attrib,greeks_rights):
            bsm = pricer.BlackScholes(R,r,q,s)
            rs[g][:,leg] = getattr(bsm,a)(S,K,T)
        rs["CHS"][:,leg] = pricer.halfspread(K,rs["CH0"][:,leg],rs["CH1"][:,leg],rs["CKH"][:,leg],"C")
        rs["PHS"][:,leg] = pricer.halfspread(K,rs["PH0"][:,leg],rs["PH1"][:,leg],rs["PKH"][:,leg],"P")
        
        if leg == 0:
            rs["CER"][:,0] = rs["RCMD"][:,0] - rs["CPX"][:,0]
            rs["PER"][:,0] = rs["RPMD"][:,0] - rs["PPX"][:,0]
            rs["VER"][:,0] = rs["RVOL"][:,0] - rs["VOL"][:,0]
            rs["EED"][:,0] = pricer.eed(rs["RCMD"][:,0],rs["RPMD"][:,0],S,K,r,q,T/base)
            
        self.rs = rs
    
    def build_trm(self,ex_jump=True):
        self.prg_trm.val += 0.1
        cols = ["SPT","CDD","BOR","DIV","FRT","YLD","VOL","CH0","CH1","CKH",
                "PH0","PH1","PKH","CPX","CEP","CHS","PPX","PEP","PHS"] 
        opt,stk = self.join_search(cols)

        x = np.zeros((opt.shape[0],1))                                          #days passed
        st = np.hstack([x,opt])                                                 # Expanding dimension for broadcasting delta vals across the depth
        st = np.repeat(st[np.newaxis,:,:],self.days_pass.shape[0],axis=0)
        st[:,:,0] = np.repeat(self.days_pass[:,np.newaxis],st.shape[1],axis=1) 
        st[:,:,8] = st[:,:,8] - st[:,:,0]                                       #CDD = CDD - days passed
        st[:,:,6] = st[:,:,6] - st[:,:,0]                                       #Tenor = tenor - days passed
        st = st.reshape((st.shape[0]*st.shape[1]),st.shape[2]); self.prg_trm.val += 0.1

        head = ["DPS","SIZ","RGT","LEG","IDX","STR","TNR"] + cols
        dtype = np.dtype([(x,float) for x in head])
        st = struct(st,dtype=dtype)
        
        o = np.where(st["LEG"]==0)[0]                                           #open idx
        c = np.where(st["LEG"]==1)[0]                                           #cls idx
        
        st["DIV"][np.where((st["CDD"]<0)&(st["LEG"]==1))[0]] = 0
        if ex_jump==True:
            st["SPT"][np.where((st["CDD"]<0)&(st["LEG"]==1))[0]] -= st["DIV"][0]
        st["CDD"] = st["CDD"]/base
        st["TNR"] = st["TNR"]/base
        
        qargs = (st["SPT"][c],st["TNR"][c],st["DIV"][c],st["CDD"][c],st["FRT"][c],)
        st["YLD"][c] = (pricer.dsc2yld(*qargs) + st["BOR"][c])  #to adjust borrow
        opx = np.where(st["RGT"][o]==1,st["CPX"][o],st["PPX"][o]) * -st["SIZ"][o]
        oep = np.where(st["RGT"][o]==1,st["CEP"][o],st["PEP"][o]) * -st["SIZ"][o]
        ohs = np.where(st["RGT"][o]==1,st["CHS"][o],st["PHS"][o]) * -abs(st["SIZ"][o]); self.prg_trm.val += 0.1

        cargs = ("C",st["SPT"][c],st["STR"][c],st["TNR"][c],st["VOL"][c],st["FRT"][c],
                      st["YLD"][c],st["CH0"][c],st["CH1"][c],st["CKH"][c])
        pargs = ("P",st["SPT"][c],st["STR"][c],st["TNR"][c],st["VOL"][c],st["FRT"][c],
                      st["YLD"][c],st["PH0"][c],st["PH1"][c],st["PKH"][c])
        cpx,cep = np.where(                                                     #Calc close px
                      st["TNR"][c]>0,
                      (np.where(                              
                               st["RGT"][c]==1,
                               pricer.ThreadPool(*cargs).run(),
                               pricer.ThreadPool(*pargs).run()
                               ) * -st["SIZ"][c]),
                      0); self.prg_trm.val += 0.1
        chs = np.where(                                                         #Calc close spread cost
                      st["TNR"][c]>0,
                      (np.where(                                  
                               st["RGT"][c]==1,
                               pricer.halfspread(st["STR"][c],st["CH0"][c],
                                                 st["CH1"][c],st["CKH"][c],"C"),
                               pricer.halfspread(st["STR"][c],st["PH0"][c],
                                                 st["PH1"][c],st["PKH"][c],"P")
                               ) * -abs(st["SIZ"][c])),
                      0); self.prg_trm.val += 0.1              
        cpf = np.where(                                                         #Calc payoff
                      st["TNR"][c]>0,
                      (np.where(                                  
                               st["RGT"][c]==1,                  
                               np.maximum(st["SPT"][c]-st["STR"][c],0),
                               np.maximum(st["STR"][c]-st["SPT"][c],0)
                               ) * -st["SIZ"][c]),
                      0); self.prg_trm.val += 0.1
        cpx = np.nan_to_num(cpx)
        cep = np.nan_to_num(cep)
        chs = np.nan_to_num(chs)
                                                                                #New array, expand back to higher dimension by pct move
        liq = np.stack([opx,oep,ohs,cpx,cep,chs],axis=1)                        #Liquidation
        liq = np.reshape(liq,(self.days_pass.shape[0],int(opt.shape[0]/2),liq.shape[1]))
        exr = np.stack([opx,oep,ohs,cpf],axis=1)                                #Exercise
        exr = np.reshape(exr,(self.days_pass.shape[0],int(opt.shape[0]/2),exr.shape[1])); self.prg_trm.val += 0.1

        if len(stk) != 0:                                                       #Spot returns, to add funding to short
            opn_size = stk[np.where(stk[:,1]==0)[0],4]
            cls_size = stk[np.where(stk[:,1]==1)[0],4] 
            spr = np.zeros((self.days_pass.shape[0],2))                         #2 cols: open & close cash flows
            spr[:,0] = (self.get_raw(0,"SPT",0)[0] * -opn_size)
            spr[:,1] = (self.get_raw(0,"SPT",1)[0] * -cls_size)
            totspr = spr[:,0] + spr[:,1]
        else:                        
            totspr = np.zeros((self.days_pass.shape[0],)) 
        ns = self.ns; self.prg_trm.val += 0.1
        liq = np.sum(liq,axis=1)                                                #Liquidation returns
        exr = np.sum(exr,axis=1)                                                #Exercise returns
        ns["LQPX"] = liq[:,0] + liq[:,3]
        ns["LQEP"] = liq[:,1] + liq[:,4]
        ns["LQHS"] = liq[:,2] + liq[:,5]
        ns["EXPX"] = exr[:,0] + exr[:,3]
        ns["EXEP"] = exr[:,1]
        ns["EXHS"] = exr[:,2]
        ns["SPTR"] = totspr
        self.ns = ns; self.prg_trm.val += 0.1  
        
    def build_pay(self):
        self.prg_pay.val += 0.1
        cols = ["CDD","BOR","DIV","FRT","YLD","VOL","CH0","CH1","CKH",
                "PH0","PH1","PKH","CPX","CEP","CHS","PPX","PEP","PHS"] 
        opt,stk = self.join_search(cols)

        x = np.zeros((opt.shape[0],2))                                          #pct move, spot
        st = np.hstack([x,opt])                                                 #Expanding dimension for broadcasting delta vals across the depth
        st = np.repeat(st[np.newaxis,:,:],self.pct_moves.shape[0],axis=0)
        st[:,:,0] = np.repeat(self.pct_moves[:,np.newaxis],st.shape[1],axis=1) 
        st[:,:,1] = self.get_raw(0,"SPT",0)[0] * (1 + st[:,:,0])                #Scls = Sopn * (1 + d%)
        st = st.reshape((st.shape[0]*st.shape[1]),st.shape[2]); self.prg_pay.val += 0.1

        head = ["PCT","SPT","SIZ","RGT","LEG","IDX","STR","TNR"] + cols
        dtype = np.dtype([(x,float) for x in head])
        st = struct(st,dtype=dtype)
        
        o = np.where(st["LEG"]==0)[0]                                           #open idx
        c = np.where(st["LEG"]==1)[0]                                           #cls idx
        
        st["CDD"] = st["CDD"]/base
        st["TNR"] = st["TNR"]/base
        st["CKH"][c] = st["CKH"][c] * (1 + st["PCT"][c])
        st["PKH"][c] = st["PKH"][c] * (1 + st["PCT"][c])
        
        qargs = (st["SPT"][c],st["TNR"][c],st["DIV"][c],st["CDD"][c],st["FRT"][c],)
        st["YLD"][c] = (pricer.dsc2yld(*qargs) + st["BOR"][c])                  #to adjust borrow
        opx = np.where(st["RGT"][o]==1,st["CPX"][o],st["PPX"][o]) * -st["SIZ"][o]
        oep = np.where(st["RGT"][o]==1,st["CEP"][o],st["PEP"][o]) * -st["SIZ"][o]
        ohs = np.where(st["RGT"][o]==1,st["CHS"][o],st["PHS"][o]) * -abs(st["SIZ"][o]); self.prg_pay.val += 0.1

        cargs = ("C",st["SPT"][c],st["STR"][c],st["TNR"][c],st["VOL"][c],st["FRT"][c],
                     st["YLD"][c],st["CH0"][c],st["CH1"][c],st["CKH"][c])
        pargs = ("P",st["SPT"][c],st["STR"][c],st["TNR"][c],st["VOL"][c],st["FRT"][c],
                     st["YLD"][c],st["PH0"][c],st["PH1"][c],st["PKH"][c])
        cpx,cep = np.where(                                                     #Calc close px
                          st["RGT"][c]==1,
                          pricer.ThreadPool(*cargs).run(),
                          pricer.ThreadPool(*pargs).run()
                        ) * -st["SIZ"][c]; self.prg_pay.val += 0.1
        chs = np.where(                                                         #Calc close spread cost
                      st["RGT"][c]==1,
                      pricer.halfspread(st["STR"][c],st["CH0"][c],st["CH1"][c],st["CKH"][c],"C"),
                      pricer.halfspread(st["STR"][c],st["PH0"][c],st["PH1"][c],st["PKH"][c],"P")
                      ) * -abs(st["SIZ"][c]); self.prg_pay.val += 0.1
        cpf = np.where(                                                         #Calc payoff
                        st["RGT"][c]==1,                  
                        np.maximum(st["SPT"][c]-st["STR"][c],0),
                        np.maximum(st["STR"][c]-st["SPT"][c],0)
                      ) * -st["SIZ"][c]; self.prg_pay.val += 0.1
                                                                                #New array, expand back to higher dimension by pct move
        liq = np.stack([opx,oep,ohs,cpx,cep,chs],axis=1)                        #Liquidation
        liq = np.reshape(liq,(self.pct_moves.shape[0],int(opt.shape[0]/2),liq.shape[1]))
        exr = np.stack([opx,oep,ohs,cpf],axis=1)                                #Exercise
        exr = np.reshape(exr,(self.pct_moves.shape[0],int(opt.shape[0]/2),exr.shape[1])); self.prg_pay.val += 0.1

        if len(stk) != 0:                                                       #Spot returns, to add funding to short
            opn_size = stk[np.where(stk[:,1]==0)[0],4]
            cls_size = stk[np.where(stk[:,1]==1)[0],4] 
            spr = np.zeros((self.pct_moves.shape[0],2))                         #2 cols: open & close cash flows
            spr[:,0] = (self.get_raw(0,"SPT",0)[0] * -opn_size)
            spr[:,1] = (self.get_raw(0,"SPT",0)[0] * (1 + self.pct_moves) * -cls_size)
            totspr = spr[:,0] + spr[:,1]
        else:                        
            totspr = np.zeros((self.pct_moves.shape[0],)) 
        ps = self.ps; self.prg_pay.val += 0.1
        liq = np.sum(liq,axis=1)                                                #Liquidation returns
        exr = np.sum(exr,axis=1)                                                #Exercise returns
        ps["LQPX"] = liq[:,0] + liq[:,3]
        ps["LQEP"] = liq[:,1] + liq[:,4]
        ps["LQHS"] = liq[:,2] + liq[:,5]
        ps["EXPX"] = exr[:,0] + exr[:,3]
        ps["EXEP"] = exr[:,1]
        ps["EXHS"] = exr[:,2]
        ps["SPTR"] = totspr
        self.ps = ps; self.prg_pay.val += 0.1
        
#------------------------------------------------------------------------------------------        
def htmlstr(font_size,text):
    return f"<font weight=100><font color='lightgray'><font size={font_size}>{text}</font>"

def create_grid(head_vals,cell_vals,head_frmt,cell_frmt,width,height,head_height,cell_height,l,r,b,t,header_fill_colour=c.lightgray,header_font_colour=c.white,cell_font_colour=c.white):
    grid = go.FigureWidget(data=[go.Table(
            header=dict(
                values=head_vals,
                height=head_height,
                font=dict(size=12,color=header_font_colour),
                format=head_frmt,
                line_color=c.darkgray,
                fill_color=header_fill_colour,
                align=["left"]*len(head_vals)),
            cells=dict(
                values=cell_vals,
                height=cell_height,
                font=dict(size=12,color=cell_font_colour),
                format=cell_frmt,
                line_color=c.darkgray,
                fill_color=c.gray,
                align=["left"]*len(cell_vals),))],
            layout=go.Layout(
                margin=go.layout.Margin(l=l,r=r,b=b,t=t,),
                paper_bgcolor=c.transparent,
                plot_bgcolor=c.transparent,
                template=c.template,
                height=height,
                width=width))
    return grid

def create_figure(xtitle,ytitle,legend_title,width,height,l,r,b,t,yzeroline=False):
    fig = go.FigureWidget(layout=go.Layout(
                margin=go.layout.Margin(l=l,r=r,b=b,t=t,),
                xaxis_title=xtitle,
                yaxis_title=ytitle,
                legend_title=legend_title,
                template=c.template,
                paper_bgcolor=c.transparent,
                plot_bgcolor=c.transparent,
                height=height,
                width=width))
    fig.update_xaxes(showgrid=False,zeroline=False,showline=True,linewidth=1,
                     linecolor=c.lightgray)
    fig.update_yaxes(showgrid=False,zeroline=yzeroline,showline=True,linewidth=1,
                     linecolor=c.lightgray,zerolinecolor=c.lightgray,zerolinewidth=1)
    return fig

def add_trace(fig,x,y,name,group,mode,colour,legend=True,marker_size=4,visible=True):
    fig.add_trace(go.Scatter(
            x = x, y = y, 
            visible = visible,
            mode = mode, 
            name = name,
            legendgroup = group,
            showlegend = legend,
            line = dict(
                        width=1,
                        color=colour
                       ),
            marker = dict(
                          size=[marker_size]*len(x),
                          color=[colour]*len(x)
                         ),
            ))
    
def factor_slider(minv,maxv,step,val=0):
    return widgets.BoundedFloatText(value=val,min=minv,max=maxv,step=step,layout=widgets.Layout(width='auto'),disabled=False)

def add_button(desc):
    return widgets.Button(description=desc,layout=widgets.Layout(width='auto')) 

def add_toggle(desc):
    return widgets.ToggleButton(description=desc,layout=widgets.Layout(width='auto'),disabled=False) 

def add_dropdown(options,val):
    return widgets.Dropdown(options=options,value=val,disabled=False,layout=widgets.Layout(width='auto'))

def add_progress_bar():
    return widgets.FloatProgress(value=0.0,min=0.0,max=1.0,layout=widgets.Layout(width='auto',height='auto'))

#------------------------------------------------------------------------------------------
class LoadData:
    def __init__(self,symbol,qdate,qtime):
        self.symbol = symbol
        self.qdate = qdate
        self.qtime = qtime

    def start(self,loc=0,scale=0): #Return here instead of post& get -> not efficient
        prg_load.val += 0.1
        
        params,errors = calibrator.surface(self.symbol,self.qdate,self.qtime,errors=True,post=False,loc=loc,scale=scale)
        self.errors = errors
        prg_load.val += 0.1
        
        k = np.hstack([-np.linspace(1,0,101)[:-1],np.linspace(0,0.7,71)])
        vols = query.vols(vol_cols,["Tenor","Strike"],self.symbol,self.qdate,self.qtime).set_index("Tenor")
        if vols.empty:
            print("No data returned for the selected date")
            prg_load.val = 0
            return
        prg_load.val += 0.1
        
        return ChainStruct(params,vols,k)
