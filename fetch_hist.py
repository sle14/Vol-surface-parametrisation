from datetime import datetime
from time import sleep
import source_hist
import numpy as np
import query
import utils
import time

cout = utils.log(__file__,__name__)
np.seterr(divide='ignore')
np.warnings.filterwarnings('ignore')
base = 253
#---------------------------------------------------------------------------------------------------------
def main(qdate,symbols):
    for symbol in symbols:
        start_time = time.time()
        
        weekday = 5 if symbol != "NDX" else 4
        req = source_hist.QuotesReq(qdate,symbol,"USD","1 D")
        
        stk = query.stk_remainder(symbol,qdate)
        while stk.empty == True: 
            cout.info(f"{qdate} {symbol} - requesting spot quotes")
            req.equities()
            stk = query.stk_remainder(symbol,qdate)
        else:
            cout.info(f"{qdate} {symbol} - stk table already populated") 
        
        opt = query.opt_remainder(3,weekday,symbol,"USD",qdate,14)
        while opt.empty == False:
            cout.info(f"{qdate} {symbol} - requesting {len(opt.index)} contract quotes")
            for i in opt.index:
                req.options(opt["Expiry"].iloc[i],opt["Strike"].iloc[i],opt["Type"].iloc[i])
            opt = query.opt_remainder(3,weekday,symbol,"USD",qdate,14)
        else:
            cout.info(f"{qdate} {symbol} - opt table already populated")
        
        elapsed = round((time.time()-start_time)/60,3)
        cout.info(f"{qdate} {symbol} - elapsed {elapsed} mins")
        
#---------------------------------------------------------------------------------------------------------        
symbols = query.get("Static","select distinct Symbol from dbo.chains")["Symbol"].sort_values().to_list()

qdate = "16/11/2020"

# qdate = input("Select trade date in dd/mm/yyyy format: ")

try:
    for q in ["16/11/2020","17/11/2020"]:
        main(q,symbols)
    cout.terminate()
except:
    cout.error("Error")

