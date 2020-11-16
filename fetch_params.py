from numpy import log,sqrt,exp,inf,nan,pi
import matplotlib.pyplot as plt 
from datetime import datetime
import pandas as pd
import numpy as np
import calibrator
import pricer
import utils
import query
import time

cout = utils.log(__file__,__name__)
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')
base = 253
#---------------------------------------------------------------------------------------------------------
# qdate = "06/11/2020"
# replace = True

qdate = input("Select trade date in dd/mm/yyyy format: ")
replace = input("Replace matching instance of params by Symbol & Date? ")

q = "select distinct Symbol from dbo.chains" #sort out index requests for spot
symbols = query.get("Static",q)["Symbol"].sort_values().to_list()

try:
    for symbol in symbols:
        while True:
            
            #Check vols are populated
            q = f"select distinct Time,Tenor,Strike from dbo.[{symbol}] where Date = convert(datetime,'{qdate}',103) order by Tenor,Strike"
            vols = query.get("Vols",q)
            quotes = query.front_series(3,symbol,"USD",qdate)[["Time","Tenor","Strike"]].drop_duplicates()
            if len(vols) != len(quotes) or (vols.empty and quotes.empty):
                cout.warn(f"{symbol} - missing vols from the selected date & symbol, unable to calibrate") 
                break
                
            #Check if table exist and params are populated
            tables = query.get("Params","select count(*) from information_schema.tables where table_name = 'main'").iloc[0,0]
            if tables == 1:   
                q = f"select distinct Symbol,Date from dbo.main where Symbol = '{symbol}' and Date = convert(datetime,'{qdate}',103)"
                params = query.get("Params",q)
                if params.empty == False and replace == False: 
                    cout.info(f"{symbol} - params already populated, moving on to next") 
                    break
                elif params.empty == False and replace == True: 
                    cout.warn(f"{symbol} - params already populated, cleaning up")
                    q = f"delete from dbo.main where Symbol = '{symbol}' and Date = convert(datetime,'{qdate}',103)"
                    query.execute("Params",q)
                else:
                    cout.info(f"{symbol} - params are empty for the selected date & symbol")     
            
            #Main
            cout.info(f"{symbol} - calibrating vols for slices")
            start_time = time.time()
            df = pd.DataFrame()
            
            for qtime in query.trading_times(qdate,symbol):
                dfx,eps,n = calibrator.surface(symbol,qdate,qtime,errors=True,n=True)
                dfx["N"] = n                                                           #Sample size
                dfx["E"] = eps[-1]/n                                                   #Mean error
                df = pd.concat([df,dfx],axis=0)  
        
            query.post(df,"Params","main","append")
    
            elapsed = round((time.time()-start_time)/60,3)
            cout.info(f"{symbol} - elapsed {elapsed} mins")
            break
        
    cout.terminate()
except:
    cout.error("Error")
#---------------------------------------------------------------------------------------------------------    

#add option for mass requests between dates for qdate
#put all checks into functions
#add query func but for params, make it as vols one