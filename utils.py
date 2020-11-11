from numpy.lib.recfunctions import unstructured_to_structured as struct
from numpy.lib.recfunctions import structured_to_unstructured as unstruct
from collections.abc import Iterable
from itertools import product
from collections import deque
from threading import Timer
import pandas as pd
import collections
import numpy as np
import logging
import time
import os

class log:
    def __init__(self,file:__file__,name:__name__,disp=True):
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logger_name = os.path.basename(file)
        logger_name = os.path.splitext(logger_name)[0]
        logger_path = f"{os.path.dirname(os.path.realpath(__file__))}\logs"
        if not os.path.exists(logger_path): os.makedirs(logger_path)
        logging.basicConfig(filename=f"{logger_path}\{logger_name}.log",
                            format='%(levelname)s:%(message)s',level=logging.DEBUG)
        self.logger = logging.getLogger(name)
        self.disp = disp

    def warn(self,text):
        text = f" {time.strftime('%H:%M:%S')} > {text}"
        if self.disp == True: print(text)
        self.logger.warn(text,exc_info=True)
    
    def error(self,text):
        text = f" {time.strftime('%H:%M:%S')} > {text}"
        if self.disp == True: print(text)
        self.logger.error(text,exc_info=True)
        self.terminate()
        
    def info(self,text):
        text = f" {time.strftime('%H:%M:%S')} > {text}"
        if self.disp == True: print(text)
        self.logger.info(text)
        
    def terminate(self):
        self.logger.info(f" {time.strftime('%H:%M:%S')} > Terminated")
        handlers = self.logger.handlers[:]
        for handler in handlers: 
            handler.close()
            self.logger.removeHandler(handler)

cout = log(__file__,__name__,disp=False)
#------------------------------------------------------------------------------------       
class watchdog:
    def __init__(self,t,f):
        cout.info("Initing watchdog")
        self.f = f
        self.t = Timer(t,f)
        self.t.start()
        
    def reset(self,t):
        cout.info("Resetting watchdog")
        self.t.cancel()
        self.__init__(t,self.f)
        
    def terminate(self):
        cout.warn("Terminating watchdog")
        self.t.cancel()
#------------------------------------------------------------------------------------
cols_dct = {
            "Date":         int,
            "Time":         int,
            "Expiry":       int,
            "Group":        int,
            "Tenor":        int,
            "Strike":       float,
            
            "Rate":         float,
            "CumDivDays":   float,
            "Div":          float,
            
            "Moneyness":    float,
            
            "SpotMid":      float,
            "SpotBid":      float,
            "SpotAsk":      float,
            "SpotSpread":   float,
            
            "CallMid":      float,
            "CallBid":      float,
            "CallAsk":      float,
            "CallSpread":   float,
            
            "PutMid":       float,
            "PutBid":       float,
            "PutAsk":       float,
            "PutSpread":    float,
            
            "ImpBor":       float,
            "ImpBorDiv":    float,
            "CallH0":       float,
            "CallH1":       float,
            "CallKh":       float,
            "PutH0":        float,
            "PutH1":        float,
            "PutKh":        float,
            "Forward":      float,
            "LogStrike":    float,
            "CallVol":      float,
            "CallEEP":      float,
            "PutVol":       float,
            "PutEEP":       float,
            "RawVol":       float,
            "SmtVol":       float,
            "TotAtmfVar":   float,   
           }

prm_cols = [
            "BBG", #Ticker
            "DAT", #Date
            "TIM", #Time
            "TNR", #Tenor
            "SPT", #Spot px
            "FWD", #Forward
            "CDD", #Cum-div days
            "DIV", #Cash div
            "BOR", #Imp borrow
            "FRT", #Funding rate
            "CH0", #Call spread min
            "CH1", #Call spread slope
            "CKH", #Call spread strike
            "PH0", #Put spread min
            "PH1", #Put spread slope
            "PKH", #Put spread strike
            "VAR", #AMTF total variance
            
            "ATM", #ATMF vol
            "SKW", #ATMF skew
            "KRT", #ATMF kurt            
            "PHI", #Convexity = phi
            "RHO", #Skew term = rho
           ]

def pack(df):
    cols = df.columns.tolist()
    dtype = [(col,cols_dct[col]) for col in cols]
    if "Date" in cols: df["Date"] = to_unix(df["Date"].to_numpy())
    if "Expiry" in cols: df["Expiry"] = to_unix(df["Expiry"].to_numpy())
    return np.array(df.to_records(index=False),dtype=dtype)

def unpack(arr,date=None,raw=True,symbol=None):
    if raw == True:
        df = pd.DataFrame(arr,columns=list(cols_dct.keys()))
        df["Date"] = to_date(df["Date"].to_numpy())
        df["Expiry"] = to_date(df["Expiry"].to_numpy())
        for x in ["Time","Group","Tenor","CumDivDays"]:
            df[x] = df[x].astype(int)
        return df
    else:
        df = pd.DataFrame(arr,columns=prm_cols[2:])
        df["DAT"] = pd.to_datetime(date,format="%d/%m/%Y")
        df["BBG"] = symbol
        return df.reindex(columns=["BBG","DAT"]+df.columns[:-2].tolist())        

#------------------------------------------------------------------------------------
def apply(f,returns,st,cat_cols=(),args=(),asarray=False,fill=False,diff=False):
    assert (returns != 1 or diff != True), "can't handle 1 return and different dimensions -> set diff to False"
    
    if returns > 1:
        deq = [deque() for i in range(returns)]
    else:
        deq = deque()
        
    if asarray == False: 
        args = unstruct(st[args])
    else:
        args = np.vstack(args).T
        
    items = [np.unique(st[i]) for i in cat_cols]
    for x in product(*items):
        cout.info(f"Categories: {x}")
        
        cond = [st[j] == x[i] for j,i in zip(cat_cols,range(len(cat_cols)))]
        idx = np.select([np.logical_and.reduce(cond)],[True],False)
        
        fargs = args[idx,:].T
        ret = list(f(*fargs))
        
        assert isinstance(ret,Iterable) != False, "arg func is not returning iterable"
        assert diff == False or len(ret) == returns, f"arg func returns {len(ret)} vals, but expected {returns}"

        if fill == True:
            if returns > 1:
                for i in range(returns):
                    ret[i] = np.resize(ret[i], fargs.shape[1])
            else:
                ret = np.resize(ret, fargs.shape[1])
        
        if returns > 1:    
            for i in range(returns): 
                deq[i].append(ret[i])
        else:
            deq.append(ret)
  
    if returns == 1:
        ret = np.hstack(list(deq)).T
        
    elif returns > 1 and diff == False and fill == False:
        ret = np.vstack(list(deq)).T
        
    elif returns > 1 and diff == False and fill == True:
        ret = [list(flatten(d)) for d in deq]
        ret = np.asarray(ret).T
        
    elif returns > 1 and diff == True:
        ret = np.asarray(deq).T
        ret = [np.hstack(ret[:,i]) for i in range(returns)]

    return ret
   
def restack(from_,to_,tile=False):
    if tile == False:
        reps = int(to_.shape[0]/from_.shape[0])
        ret = np.repeat(from_,reps,axis=0)
    else:
        ret = np.tile(from_,to_.shape[0])
    return ret

def select(st,cols=()):
    arr = np.hsplit(unstruct(st[cols]),len(cols))
    arr = [np.squeeze(x) for x in arr]
    return arr

#------------------------------------------------------------------------------------
def to_unix(arr):
    return pd.to_datetime(arr).astype(int) / 10**9

def to_date(arr):
    return pd.to_datetime(arr,unit="s")

def closest(lst,x): 
    return lst[min(range(len(lst)),key=lambda i:abs(lst[i]-x))] 

def intersect2d(X,Y):
    X = np.tile(X[:,:,None], (1, 1, Y.shape[0]) )
    Y = np.swapaxes(Y[:,:,None], 0, 2)
    Y = np.tile(Y, (X.shape[0], 1, 1))
    eq = np.all(np.equal(X, Y), axis = 1)
    eq = np.any(eq, axis = 1)
    return np.nonzero(eq)[0]

def unique_nosort(arr):
    arr,idx = np.unique(arr,return_index=True)
    ret = arr[np.argsort(idx)]
    return ret
    
def sort_array(arr,cols:tuple):
    arr = arr[arr[:,cols[0]].argsort()]                 
    arr = arr[arr[:,cols[1]].argsort(kind='mergesort')] 
    arr = arr[arr[:,cols[2]].argsort(kind='mergesort')] 
    if len(cols)==4: arr = arr[arr[:,cols[3]].argsort(kind='mergesort')] 
    return arr
    
def rshift(arr,n):
    sft_arr = np.empty_like(arr)
    sft_arr[:n] = arr[-n:]
    sft_arr[n:] = arr[:-n]
    return sft_arr

def flatten(lst):
    for i in lst:
        if isinstance(i,collections.Iterable) and not isinstance(i,(str,bytes)):
            yield from flatten(i)
        else:
            yield i

def stretch(lst,n):
    x = np.arange(n,dtype=int)
    arr = np.full_like(x,np.nan,dtype=np.double)
    arr[:len(lst)] = lst
    return arr

def listgroup(df,agg_col):
    return [df[df.index==t][agg_col].to_numpy() for t in df.index.unique()]

def dropna(arr):
    return arr[~np.isnan(arr)]