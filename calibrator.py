from numpy.lib.recfunctions import unstructured_to_structured as struct
from numpy.lib.recfunctions import structured_to_unstructured as unstruct
import numpy as np
import pricer
import utils
import query

np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')
base = 253

def surface(symbol,qdate,qtime=None,errors=False,n=False,loc=0,scale=0):
    df = query.vols(["*"],["Date","Time","Tenor","Strike"],symbol,qdate,qtime)
    st = utils.pack(df)
    del df;
    
    #----------------------------------------------------------------------------------
    f = lambda LST: pricer.norm_weights(LST,loc,scale)
    wgt = utils.apply(f,1,st,["Time","Tenor"],["LogStrike"]) #retrive weights vector

    sv = pricer.SSVI(errors)
    f = lambda LST,VAR,TNR,VOL: sv.calibrate(LST,VAR,TNR,VOL,wgt)
    
    if errors == True:
        prm = utils.apply(f,6,st,["Time"],["LogStrike","TotAtmfVar","Tenor","SmtVol"],diff=True,fill=False)
        eps = prm[-1]
        prm = np.asarray(prm[:-1]).T
    else:
        prm = utils.apply(f,5,st,["Time"],["LogStrike","TotAtmfVar","Tenor","SmtVol"],fill=False)

    #----------------------------------------------------------------------------------
    reduced = ["Time","Group","Tenor","SpotMid","Forward","CumDivDays","Div","ImpBor",
               "Rate","CallH0","CallH1","CallKh","PutH0","PutH1","PutKh","TotAtmfVar"]
    st = utils.unique_nosort(st[reduced])
    
    prm = utils.restack(prm,st["TotAtmfVar"],tile=False)
    
    rho = sv.correlation(st["TotAtmfVar"], prm[:,0], prm[:,1], prm[:,2])
    phi = sv.power_law(st["TotAtmfVar"], prm[:,3], prm[:,4])
    atm,skw,krt = sv.raw2jw(st["Tenor"]/base,st["TotAtmfVar"],phi,rho)
    
    arr = unstruct(st)
    arr = np.column_stack([arr,atm,skw,krt,phi,rho])
    df = utils.unpack(arr,qdate,raw=False,symbol=symbol)
    
    if errors == True and n == False:
        return df,eps
    elif errors == False and n == True:
        return df,n
    elif errors == True and n == True:
        return df,eps,n
    else:
        return df