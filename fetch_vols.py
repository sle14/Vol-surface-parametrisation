from numpy.lib.recfunctions import unstructured_to_structured as struct
from numpy.lib.recfunctions import structured_to_unstructured as unstruct
from time import sleep
import numpy as np
import pricer
import utils
import query
import time

cout = utils.log(__file__,__name__)
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')
base = 253
#---------------------------------------------------------------------------------------------------------
def checks(qdate,symbol,quotes,weekday):
    #Check if quotes are populated
    if quotes.empty: 
        print(f"{qdate} {symbol} - Nothing returned on quotes, are rates populated?")
        return 0
    else:
        while True:
            remainder = query.opt_remainder(3,weekday,symbol,"USD",qdate,14)
            if remainder.empty: break
            left = len(remainder)
            cout.warn(f"{qdate} {symbol} - {left} quotes missing, waiting for fetch to populate") 
            sleep(60)

    #Check if table exist and vols are populated
    tables = query.get("Vols",f"select count(*) from information_schema.tables where table_name = '{symbol}'").iloc[0,0]
    if tables == 1:
        q = f"select distinct Time,Tenor,Strike from dbo.[{symbol}] where Date = convert(datetime,'{qdate}',103) order by Tenor,Strike"
        vols = query.get("Vols",q)
        quotes = quotes[["Time","Tenor","Strike"]].drop_duplicates()
        if len(vols) == len(quotes):
            cout.info(f"{qdate} {symbol} - vols already populated, moving on to next") 
            return 0
        elif len(vols) > 0:
            cout.warn(f"{qdate} {symbol} - vols are partially populated, cleaning up")
            q = f"delete from dbo.[{symbol}] where Date = convert(datetime,'{qdate}',103)"
            query.execute("Vols",q)
        else:
            cout.info(f"{qdate} {symbol} - vols are empty for the selected date & symbol") 

def main(qdate,symbols,qtime=None):
    for symbol in symbols:
        start_time = time.time()
        
        weekday = 5 if symbol != "NDX" else 4
        quotes = query.front_series(3,weekday,symbol,"USD",qdate,qtime)
        
        if checks(qdate,symbol,quotes,weekday) == 0: continue
        
        cout.info(f"{qdate} {symbol} - initing vol fetch")
        st = utils.pack(quotes)
        
        f = pricer.impdivsborrows
        yld = utils.apply(f,2,st,["Time","Tenor"],["Moneyness","SpotMid","Strike","Tenor","CallBid","PutAsk","Rate","Div","CumDivDays"],fill=True)
        
        f = lambda STR,SPT,SPR: pricer.fit_spread(STR,SPT,SPR,"C")
        chs = utils.apply(f,3,st,["Time","Tenor"],["Strike","SpotMid","CallSpread"],fill=True)
        
        f = lambda STR,SPT,SPR: pricer.fit_spread(STR,SPT,SPR,"P")
        phs = utils.apply(f,3,st,["Time","Tenor"],["Strike","SpotMid","PutSpread"],fill=True)
        
        f = lambda TNR,SPT,FRT,CDD,DIV: pricer.forward(TNR,SPT,FRT,yld[:,0],CDD,DIV)
        args = utils.select(st,["Tenor","SpotMid","Rate","CumDivDays","Div"])
        fwd = f(*args)
        lst = np.log(st["Strike"]/fwd)
        
        f = lambda CPX,SPT,STR,TNR,FRT: pricer.Vol("C",CPX,SPT,STR,TNR,FRT,yld[:,1],chs[:,0],chs[:,1],chs[:,2]).root(0.01,4)
        args = utils.select(st,["CallMid","SpotMid","Strike","Tenor","Rate"])
        cvol,ceep = f(*args)
        
        f = lambda PPX,SPT,STR,TNR,FRT: pricer.Vol("P",PPX,SPT,STR,TNR,FRT,yld[:,1],phs[:,0],phs[:,1],phs[:,2]).root(0.01,4)
        args = utils.select(st,["PutMid","SpotMid","Strike","Tenor","Rate"])
        pvol,peep = f(*args)
        
        ceep = utils.apply(pricer.interp,1,st,["Time","Tenor"],[lst,ceep],asarray=True,fill=True)
        peep = utils.apply(pricer.interp,1,st,["Time","Tenor"],[lst,peep],asarray=True,fill=True)
        
        rvol = [(i+j)/2 if np.isnan(i)==False and np.isnan(j)==False else j if np.isnan(j)==False else i if np.isnan(i)==False else np.nan for i,j in zip(cvol,pvol)]
        svol = utils.apply(pricer.interp,1,st,["Time","Tenor"],[lst,rvol],asarray=True,fill=True)
        
        var = utils.apply(pricer.totatmfvar,1,st,["Time","Tenor"],[st["Tenor"],lst,rvol],asarray=True,fill=True)
        if len(var[np.isnan(var)]) > 0:
            var = utils.apply(pricer.totatmfvar,1,st,["Time","Tenor"],[st["Tenor"],lst,svol],asarray=True,fill=True)
        
        ar = np.column_stack([unstruct(st),yld,chs,phs,fwd,lst,cvol,ceep,pvol,peep,rvol,svol,var])
        df = utils.unpack(ar)
        
        query.post(df,"Vols",symbol,"append")
        
        elapsed = round((time.time()-start_time)/60,3)
        cout.info(f"{qdate} {symbol} - elapsed {elapsed} mins")

#---------------------------------------------------------------------------------------------------------
symbols = query.get("Static","select distinct Symbol from dbo.chains")["Symbol"].sort_values().to_list()

# qdate = "16/11/2020"
# qtime = "15:50"

# start_date = "11/11/2020"
# end_date = "19/11/2020"
# qdates = [x.strftime('%d/%m/%Y') for x in utils.drange(start_date,end_date)]

qdate = input("Select trade date in dd/mm/yyyy format: ")

try:
    main(qdate,symbols)       
    cout.terminate()
except:
    cout.error("Error")  
