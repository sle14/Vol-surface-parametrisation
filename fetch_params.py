from numpy import log,sqrt,exp,inf,nan,pi
import matplotlib.pyplot as plt 
from datetime import datetime
from time import sleep
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
def checks(qdate,symbol,quotes):
    #Check if vols are populated
    if quotes.empty: 
        print(f"{qdate} {symbol} - Nothing returned on quotes, are rates populated?")
        return 0
    else:
        while True:
            vols = query.vols(["Time","Tenor","Strike"],["Tenor","Strike"],symbol,qdate,qtime=None,distinct=True)
            if vols.empty == False: break
            cout.warn(f"{qdate} {symbol} - missing vols from the selected date & symbol, unable to calibrate, waiting for fetch to populate") 
            sleep(60)

    #Check if table exist and params are populated
    tables = query.get("Params","select count(*) from information_schema.tables where table_name = 'main'").iloc[0,0]
    if tables == 1:   
        q = f"select distinct Symbol,Date from dbo.main where Symbol = '{symbol}' and Date = convert(datetime,'{qdate}',103)"
        params = query.get("Params",q)
        if params.empty == False and replace == False: 
            cout.info(f"{qdate} {symbol} - params already populated, moving on to next") 
            return 0
        elif params.empty == False and replace == True: 
            cout.warn(f"{qdate} {symbol} - params already populated, cleaning up")
            q = f"delete from dbo.main where Symbol = '{symbol}' and Date = convert(datetime,'{qdate}',103)"
            query.execute("Params",q)
        else:
            cout.info(f"{qdate} {symbol} - params are empty for the selected date & symbol") 
            
def main(qdate,symbols):
    for symbol in symbols:
        start_time = time.time()
           
        weekday = 5 if symbol != "NDX" else 4
        quotes = query.front_series(3,weekday,symbol,"USD",qdate)[["Time","Tenor","Strike"]].drop_duplicates()
        
        if checks(qdate,symbol,quotes) == 0: continue
        
        cout.info(f"{qdate} {symbol} - calibrating vols for slices")
        df = pd.DataFrame()

        for qtime in query.trading_times(qdate,symbol):
            dfx,eps,n = calibrator.surface(symbol,qdate,qtime,errors=True,n=True)
            dfx["N"] = n                                                           #Sample size
            dfx["E"] = eps[-1]/n                                                   #Mean error
            df = pd.concat([df,dfx],axis=0)  
    
        query.post(df,"Params","main","append")

        elapsed = round((time.time()-start_time)/60,3)
        cout.info(f"{qdate} {symbol} - elapsed {elapsed} mins")
        
#---------------------------------------------------------------------------------------------------------
symbols = query.get("Static","select distinct Symbol from dbo.chains")["Symbol"].sort_values().to_list()
replace = False

qdates = ["16/11/2020"]

# start_date = input("Select start trade date in dd/mm/yyyy format: ")
# end_date = input("Select end trade date in dd/mm/yyyy format: ")
# qdates = [x.strftime('%d/%m/%Y') for x in utils.drange(start_date,end_date)]

try:
    for qdate in qdates: main(qdate,symbols)
    cout.terminate()
except:
    cout.error("Error")
    
#Sort BIIB for 06/11/2020
#add query func but for params, make it as vols one