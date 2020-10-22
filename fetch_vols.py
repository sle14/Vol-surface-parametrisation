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

# qdate = "20/10/2020"
# qtime = "14:40"

symbol = "AAL"
curr = "USD"

qdate = input("Select trade date in dd/mm/yyyy format: ")

q = "select distinct Symbol from dbo.chains where Symbol != 'FP'"
symbols = query.get("Static",q)
symbols = symbols["Symbol"].sort_values().to_list()

try:
    for symbol in symbols:
        while True:
            df = query.front_series(3,symbol,curr,qdate)
            
            #Check quotes are populated
            remainder = query.opt_remainder(3,symbol,curr,qdate)
            if remainder.empty == False: 
                left = len(remainder)
                cout.info(f"{symbol} - {left} quotes missing, waiting for fetch to populate") 
                sleep(60)
                continue
            
            #Check if table exist and vols are populated
            tables = query.get("Vols",f"select count(*) from information_schema.tables where table_name = '{symbol}'").iloc[0,0]
            if tables == 1:
                q = f"select distinct Time,Tenor,Strike from dbo.{symbol} where Date = convert(datetime,'{qdate}',103) order by Tenor,Strike"
                vols = query.get("Vols",q)
                quotes = df[["Time","Tenor","Strike"]].drop_duplicates()
                if len(vols) == len(quotes):
                    cout.info(f"{symbol} - vols already populated, moving on to next") 
                    break
                elif len(vols) > 0:
                    cout.warn(f"{symbol} - vols are partially populated, cleaning up")
                    q = f"delete from dbo.{symbol} where Date = convert(datetime,'{qdate}',103)"
                    query.execute("Vols",q)
                else:
                    cout.info(f"{symbol} - vols are empty for the selected date & symbol") 
            
            #Main
            cout.info(f"{symbol} - initing vol fetch")
            start_time = time.time()
            st = utils.pack(df)
            
            f = pricer.impdivsborrows
            yld = utils.apply(f,2,st,["Time","Tenor"],["Moneyness","SpotMid","Strike","Tenor","CallMid","PutMid","Rate","Div","CumDivDays"],fill=True)
            
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
            
            cvol = utils.apply(pricer.interp,1,st,["Time","Tenor"],[lst,cvol],asarray=True,fill=True)
            pvol = utils.apply(pricer.interp,1,st,["Time","Tenor"],[lst,pvol],asarray=True,fill=True)
            
            rvol = (cvol + pvol)/2
            svol = utils.apply(pricer.interp,1,st,["Time","Tenor"],[lst,rvol],asarray=True,fill=True)
            
            var = utils.apply(pricer.totatmfvar,1,st,["Time","Tenor"],[st["Tenor"],lst,svol],asarray=True,fill=True)
            
            ar = np.column_stack([unstruct(st),yld,chs,phs,fwd,lst,cvol,ceep,pvol,peep,rvol,svol,var])
            df = utils.unpack(ar,qdate)
            
            query.post(df,"Vols",symbol,"append")
            
            elapsed = round((time.time()-start_time)/60,3)
            cout.info(f"{symbol} - elapsed {elapsed} mins")
            break
    cout.terminate()
except:
    cout.error("Error")    