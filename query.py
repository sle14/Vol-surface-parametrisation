from datetime import date,datetime
from sqlalchemy import text
import pandas as pd
import numpy as np
import sqlalchemy
import pyodbc
import utils
import time

pyodbc.pooling = False
cout = utils.log(__file__,__name__,disp=False)
server = "DESKTOP-1HLMBUK"
driver = "SQL Server Native Client 11.0"
user = "Aigars"

def opt_stack(front_months,weekday,symbol,curr,qdate,qtime=None,opt_table=None):
    opt_table = symbol if opt_table == None else opt_table
    query = f"""
    declare @symbol varchar(60); set @symbol = '{symbol}';
    declare @curr varchar(60); set @curr = '{curr}';
    declare @qdate datetime; set @qdate = convert(datetime,'{qdate}',103);
	declare @maxqdate datetime; set @maxqdate = (select max(Expiry) as Expiry from (
		select distinct top {front_months} Expiry from Static.dbo.chains 
		where Expiry > @qdate and Symbol = @symbol and Lotsize = 100 and Currency = @curr
		and (datepart(weekday, Expiry) + @@DATEFIRST - 2) % 7 + 1 = {weekday}   -- 5 -> Friday
		and (datepart(day, Expiry) - 1) / 7 + 1 = 3                             -- 3 -> 3rd week
		order by Expiry) as sq);
    with 
    type_stack as (
        select Symbol,[Date],[Time],Expiry,cast(Strike as float) as Strike,
        sum([CBid]) as 'CallBid',sum([PBid]) as 'PutBid',
        sum([CAsk]) as 'CallAsk',sum([PAsk]) as 'PutAsk'
        from (select Symbol,[Date],[Time],Expiry,Strike,[Type],Bid,Ask,
			[Type]+'Bid' as TypeBid,[Type]+'Ask' as TypeAsk
        	from Quotes.dbo.[{symbol}] where [Date] = @qdate and Expiry <= @maxqdate 
        	and Bid > 0 and Ask > 0 and Lotsize = 100) as sq
        pivot (sum(Bid) for TypeBid in ([CBid],[PBid])) as pvt_bid
        pivot (sum(Ask) for TypeAsk in ([CAsk],[PAsk])) as pvt_ask
        group by Symbol,[Date],[Time],Expiry,Strike),
    opt_grid as (
        select *,(CallBid+CallAsk)/2 as CallMid,(PutBid+PutAsk)/2 as PutMid from type_stack
        where CallBid is not null and CallAsk is not null and PutBid is not null and PutBid is not null),
    rates_grid as (
        select [Date],Rate from (select * from Static.dbo.rates where Tenor = '3M' and Currency = @curr) as sq),
    spot_grid as (
		select Symbol,[Date],[Time],Bid,Ask from Quotes.dbo.Spot),
    divs_grid as (
        select top 1 exDate as exDate,Symbol,Amount from (select * from Static.dbo.divs 
		where exDate > @qdate and Symbol = @symbol) as sd)
    select 
        opt_grid.Date,
        opt_grid.Time,
        opt_grid.Expiry,
        dense_rank() over (order by opt_grid.Expiry)-1 as [Group],
        (datediff(dd,@qdate,Expiry) + 1) 
    - (datediff(wk,@qdate,Expiry) * 2)
    - (case when datename(dw,@qdate) = 'Sunday' then 1 else 0 end) 
    - (case when datename(dw,Expiry) = 'Saturday' then 1 else 0 end) as Tenor,
        opt_grid.Strike,
        rates_grid.Rate, 
        isnull((datediff(dd,@qdate,exDate - 1) + 1)
    - (datediff(wk,@qdate,exDate - 1) * 2)
    - (case when datename(dw,@qdate) = 'Sunday' then 1 else 0 end)
    - (case when datename(dw,exDate - 1) = 'Saturday' then 1 else 0 end),0) as CumDivDays,
        isnull(divs_grid.Amount,0) as Div,
        round(opt_grid.Strike/((spot_grid.Bid+spot_grid.Ask)/2),2) as Moneyness,
		spot_grid.Bid as SpotBid,
		spot_grid.Ask as SpotAsk,
        round((spot_grid.Bid+spot_grid.Ask)/2,4) as SpotMid,
        round(spot_grid.Ask-spot_grid.Bid,2) as SpotSpread,
        opt_grid.CallBid,
        opt_grid.CallAsk,
        opt_grid.CallMid,
        round(opt_grid.CallAsk-opt_grid.CallBid,2) as CallSpread,
        opt_grid.PutBid,
        opt_grid.PutAsk,
        opt_grid.PutMid,
        round(opt_grid.PutAsk-opt_grid.PutBid,2) as PutSpread
    from opt_grid
    join spot_grid on opt_grid.Date = spot_grid.Date and 
	opt_grid.Time = spot_grid.Time and opt_grid.Symbol = spot_grid.Symbol 
    join rates_grid on opt_grid.Date = rates_grid.Date
    left join divs_grid on opt_grid.Symbol = divs_grid.Symbol
    where opt_grid.PutAsk-opt_grid.PutBid >= 0 and opt_grid.CallAsk-opt_grid.CallBid >= 0"""
    if qtime != None: 
        qtime = int(qtime.replace(":",""))
        query += f" and opt_grid.[Time] = {qtime}"    
    query += " order by Time,Tenor,Strike"
    return query

def vols(cols,order_by,symbol,qdate=None,qtime=None,distinct=False):
    assert type(cols) is list, "cols requested must be in list format"
    assert type(order_by) is list, "order_by requested must be in list format"
    if qdate is not None: assert type(qdate) is str, "qdate has to be string in dd/mm/yyyy format" 
    if qtime is not None: assert type(qtime) is str, "qtime has to be string in hh:mm format" 
        
    q = "select"
    if distinct == True: 
        q += " distinct"
    
    if "*" not in cols: 
        q += f" {','.join('['+str(i)+']' for i in cols)}"
    else:
        q += " *"
        
    q += f" from dbo.[{symbol}]"
    
    if qdate is not None or qtime is not None: 
        q += " where "
    if qdate is not None and qtime is not None: 
        q += f" Date = convert(datetime,'{qdate}',103) and Time = {int(qtime.replace(':',''))}"
    elif qdate is not None:
        q += f" Date = convert(datetime,'{qdate}',103)"
    elif qtime is not None:
        q += f" Time = {int(qtime.replace(':',''))}"
    q += f" order by {','.join(str(i) for i in order_by)}"
    
    df = get("Vols",q)
    # if df.empty: print(f"Returned empty df on vol query with args: {cols}, {symbol}, {qdate}, {qtime}")
    return df

def params(cols,order_by,symbol=None,qdate=None,qtime=None,distinct=False):
    assert type(cols) is list, "cols requested must be in list format"
    assert type(order_by) is list, "order_by requested must be in list format"
    if qdate is not None: assert type(qdate) is str, "qdate has to be string in dd/mm/yyyy format" 
    if qtime is not None: assert type(qtime) is str, "qtime has to be string in hh:mm format" 
        
    q = "select"
    if distinct == True: 
        q += " distinct"
    
    if "*" not in cols: 
        q += f" {','.join('['+str(i)+']' for i in cols)}"
    else:
        q += " *"
        
    q += " from dbo.main"
    
    if qdate is not None or qtime is not None: 
        q += " where"
    if symbol is not None:
        q += f" Symbol = '{symbol}' and"
    if qdate is not None and qtime is not None: 
        q += f" Date = convert(datetime,'{qdate}',103) and Time = {int(qtime.replace(':',''))}"
    elif qdate is not None:
        q += f" Date = convert(datetime,'{qdate}',103)"
    elif qtime is not None:
        q += f" Time = {int(qtime.replace(':',''))}"
    q += f" order by {','.join(str(i) for i in order_by)}"
    
    df = get("Params",q)
    # if df.empty: print(f"Returned empty df on vol query with args: {cols}, {symbol}, {qdate}, {qtime}")
    return df

def front_series(front_months,weekday,symbol,curr,qdate,qtime=None,opt_table=None):
    q = opt_stack(front_months,weekday,symbol,curr,qdate,qtime,opt_table)
    df = get("Quotes",q)
    df["CumDivDays"] = np.where(df["Tenor"]<df["CumDivDays"],0,df["CumDivDays"])
    df["Div"] = np.where(df["CumDivDays"]==0,0,df["Div"])
    df = df.groupby(["Time","Tenor"]).filter(lambda x:len(x) > 2) #Drop if less than 3 occurences 
    df = df.reset_index(drop=True)
    x = df.groupby(["Time","Tenor"])["Strike"].count().unstack(1)
    x = x.columns[x.isna().any()].tolist()
    return df[~df["Tenor"].isin(x)] #Exclude tenors which are missing quotes for any particular slice

def front_static(front_months,weekday,symbol,curr,qdate,spotpx=0.,lbound_money=0.6,ubound_money=1.4):
    if spotpx == 0:
        spot = """(
    	select sum((Bid+Ask)/2)/count(Bid) as Mid from Quotes.dbo.Spot
    	where Symbol = @symbol and Date = @qdate and Currency = @curr);
        """
    else:
        spot = spotpx
    query = f"""
    declare @symbol varchar(60); set @symbol = '{symbol}';
    declare @curr varchar(60); set @curr = '{curr}';
    declare @qdate datetime; set @qdate = convert(datetime,'{qdate}',103);
    declare @spot float; set @spot = {spot}
    select * from (
    	select Expiry,convert(float,Strike) as Strike,Type
    	from Static.dbo.chains 
    	where Expiry in (
    		select distinct top {front_months} Expiry from Static.dbo.chains where Expiry > @qdate
    		and (datepart(weekday, Expiry) + @@DATEFIRST - 2) % 7 + 1 = {weekday}   -- 5 -> Friday
    		and (datepart(day, Expiry) - 1) / 7 + 1 = 3                             -- 3 -> 3rd week
    		order by Expiry)
    	and Symbol = @symbol and Lotsize = 100 and Currency = @curr) as sq
    where Strike/@spot < {ubound_money} and Strike/@spot > {lbound_money}
    order by Expiry,Strike,Type"""
    return get("Static",query)

def trading_times(qdate,symbol="AAPL"):
    if type(qdate) != str: qdate = qdate.strftime('%d/%m/%Y')
    tms = get("Vols",f"select distinct Time from dbo.[{symbol}] where Date = convert(datetime,'{qdate}',103)")["Time"].tolist()
    return [str(x)[:2]+":"+str(x)[2:] for x in list(sorted(tms))]
    
def execute(database,query):
    database=database
    eng = f"mssql+pyodbc://{user}@{server}/{database}?driver={driver}"
    engine = sqlalchemy.create_engine(eng)
    engine.execute(text(query))
    cout.info(f"Executed to {database} DB")

def get(database,query):
    eng = f"mssql+pyodbc://{user}@{server}/{database}?driver={driver}"
    engine = sqlalchemy.create_engine(eng)
    cout.info(f"Get grid from {database} DB")
    return pd.read_sql(query,con=engine)

def post(df,database,table_name,ifexists):     
    database=database
    eng = f"mssql+pyodbc://{user}@{server}/{database}?driver={driver}"
    engine = sqlalchemy.create_engine(eng)
    df.to_sql(name=table_name,con=engine,if_exists=ifexists,index=False)
    cout.info(f"Posted grid to {database} DB")

def drop_dupes(on:list,database,table_name):
    df = get(database,f"select * from dbo.[{table_name}]")
    dfc = df.drop_duplicates(on)
    dropped_rows = int(len(df.index) - len(dfc.index))
    post(dfc,database,table_name,"replace")
    print(f"Dropped {dropped_rows} dupes from {database}.dbo.{table_name}")

def drop_strikes(arr,num,exp): 
    median = (min(arr) + max(arr))/2
    xu = np.geomspace(median,max(arr),int(num/2))
    xl = min(arr) + median - np.geomspace(min(arr),median,int(num/2))
    x = np.sort(np.hstack([xl,xu[1:]]))
    x = [utils.closest(arr,i) for i in x]
    return [i if i in x else np.nan for i in arr],exp

def filter_static(df,num):
    df = df.drop(columns=["Type"]).drop_duplicates(["Expiry","Strike"])
    st = utils.pack(df)
    f = lambda arr,exp: drop_strikes(arr,num,exp)
    arr = utils.apply(f,2,st,["Expiry"],["Strike","Expiry"],fill=True)
    return arr

def opt_remainder(front_months,weekday,symbol,curr,qdate,num):
    df = front_static(front_months,weekday,symbol,curr,qdate)
    if df.empty:
        print(f"{time.strftime('%H:%M:%S')} > Nothing to return from static, is spot data populated?")
        return df
    
    #Drop strikes
    df = pd.DataFrame(filter_static(df,num),columns=["Strike","Expiry"]).dropna()
    df["Expiry"] = utils.to_date(df["Expiry"].to_numpy())
    df["Type"] = "C"
    dfx = df.copy()
    dfx["Type"] = "P"
    df = pd.concat([df,dfx],axis=0).sort_values(by=["Strike","Expiry","Type"])
    df = df.set_index(["Expiry","Type","Strike"])
    del dfx;
    
    q = f"select distinct Expiry,Type,Strike from dbo.[{symbol}] where Date = convert(datetime,'{qdate}',103)"
    try:
        opt = get("Quotes",q).set_index(["Expiry","Type","Strike"])
        opt["Exists"] = 1.
    except:
        return df.reset_index()
    
    df = pd.concat([df,opt],axis=1)
    df = df[df["Exists"].isnull()]
    df = df.reset_index().drop(columns=["Exists"])
    return df

def stk_remainder(symbol,qdate):
    q = f"select TOP 1 Date from dbo.Spot where Symbol = '{symbol}' and Date = convert(datetime,'{qdate}',103)"
    return get("Quotes",q)