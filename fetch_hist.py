import source_hist
import query
import utils

cout = utils.log(__file__,__name__)

# qdate = "26/10/2020"

curr = "USD"
window = "1 D"
qdate = input("Select trade date in dd/mm/yyyy format: ")

q = "select distinct Symbol from dbo.chains"
symbols = query.get("Static",q)
symbols = symbols["Symbol"].sort_values().to_list()

try:
    for symbol in symbols:
        req = source_hist.QuotesReq(qdate,symbol,curr,window)
        
        stk = query.stk_remainder(symbol,qdate)
        while stk.empty == True: 
            cout.info(f"{symbol} - requesting spot quotes")
            req.equities()
            stk = query.stk_remainder(symbol,qdate)
        else:
            cout.info(f"{symbol} - stk table already populated") 
        
        opt = query.opt_remainder(3,symbol,curr,qdate)
        while opt.empty == False:
            cout.info(f"{symbol} - requesting {len(opt.index)} contract quotes")
            for i in opt.index:
                req.options(opt["Expiry"].iloc[i],opt["Strike"].iloc[i],opt["Type"].iloc[i])
            opt = query.opt_remainder(3,symbol,curr,qdate)
        else:
            cout.info(f"{symbol} - opt table already populated")
    cout.terminate()
except:
    cout.error("Error")
    
# query.drop_dupes(["Date","Time","Symbol","Expiry","Strike","Type","Lotsize","Currency"],"Quotes","AMD")