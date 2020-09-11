from scipy.interpolate import PchipInterpolator
import pandas as pd
import numpy as np
import sqlalchemy
import sys

def query_sql(qdate,qexpiry):
    return f"""with 
    lot_stack as (
    	select Symbol,[Date],[Time],Expiry,Strike,[Type],
    	sum([100Bid]) as '100Bid',sum([10Bid]) as '10Bid',
    	sum([100Ask]) as '100Ask',sum([10Ask]) as '10Ask'
    	from (select Symbol,[Date],[Time],Expiry,Strike,[Type],Bid,Ask,
    		convert(varchar(10),Lotsize)+'Bid' as LotsizeBid,
    		convert(varchar(10),Lotsize)+'Ask' as LotsizeAsk
    		from dbo.options where convert(datetime,Expiry,103) < convert(datetime,'{qexpiry}',103) 
    		and [Date] = '{qdate}' and Bid != 0 and Ask != 0) as sq
    	pivot (sum(Bid) for LotsizeBid in ([100Bid],[10Bid])) as pvt_bid
    	pivot (sum(Ask) for LotsizeAsk in ([100Ask],[10Ask])) as pvt_ask
    	group by Symbol,[Date],[Time],Expiry,Strike,[Type]),
    type_stack as (
    	select Symbol,[Date],[Time],Expiry,convert(float,Strike) as Strike,
    	sum(Call100Bid) as BidAC,sum(Put100Bid) as BidAP,sum(Call10Bid) as BidEC,sum(Put10Bid) as BidEP,
    	sum(Call100Ask) as AskAC,sum(Put100Ask) as AskAP,sum(Call10Ask) as AskEC,sum(Put10Ask) as AskEP
    	from (select Symbol,[Date],[Time],Expiry,Strike,
    		[100Bid],[10Bid],[Type]+'100Bid' as Type100Bid,[Type]+'10Bid' as Type10Bid,
    		[100Ask],[10Ask],[Type]+'100Ask' as Type100Ask,[Type]+'10Ask' as Type10Ask from lot_stack) as sq
    	pivot (sum([100Bid]) for Type100Bid in ([Call100Bid],[Put100Bid])) as pvt_american_bid
    	pivot (sum([10Bid]) for Type10Bid in ([Call10Bid],[Put10Bid])) as pvt_european_bid
    	pivot (sum([100Ask]) for Type100Ask in ([Call100Ask],[Put100Ask])) as pvt_american_ask
    	pivot (sum([10Ask]) for Type10Ask in ([Call10Ask],[Put10Ask])) as pvt_european_ask
    	group by Symbol,[Date],[Time],Expiry,Strike),
    opt_grid as (
    	select *,(BidAC+AskAC)/2 as MidAC,(BidAP+AskAP)/2 as MidAP,(BidEC+AskEC)/2 as MidEC,(BidEP+AskEP)/2 as MidEP from type_stack),
    rates_grid as (
    	select [Date],Rate from (select * from dbo.rates where Tenor = '6M') as sq)
    select opt_grid.*,dbo.equities.Bid as 'BidSpot',dbo.equities.Ask as 'AskSpot',dbo.equities.Mid as 'MidSpot',rates_grid.Rate
    from opt_grid
    join dbo.equities on opt_grid.Date = dbo.equities.Date and opt_grid.Time = dbo.equities.Time 
    join rates_grid on opt_grid.Date = rates_grid.Date"""
    
def get_frame(database,query):
    server="LAPTOP-206OR7PL\\SQLEXPRESS"
    engine = sqlalchemy.create_engine('mssql+pyodbc://@'+server+'/'+
    database+'?trusted_connection=yes&driver=ODBC+Driver+13+for+SQL+Server')
    return pd.read_sql(query,con=engine)
    
def divs(df,date,expiry):
    div = get_frame("Historical","select Symbol,exDate,Amount as 'Div' from dbo.divs")
    div = div[pd.to_datetime(div["exDate"],format="%d/%m/%Y")<pd.to_datetime(expiry,format="%d/%m/%Y")]
    div = div[pd.to_datetime(div["exDate"],format="%d/%m/%Y")>pd.to_datetime(date,format="%d/%m/%Y")]
    df = df.set_index("Symbol").join(div.set_index("Symbol")).reset_index()
    return df
    
def days(df,col_a,col_b,new_col):
    for x in [col_a,col_b]: df[x] = pd.to_datetime(df[x],format="%d/%m/%Y")
    df[new_col] = (df[col_a]-df[col_b]).dt.days
    for x in [col_a,col_b]: df[x] = df[x].apply(lambda x:x.strftime("%d/%m/%Y"))
    return df
    
def opt_chain(qdate,qtime=None):   
    if not np.is_busday(pd.to_datetime(qdate,format="%d/%m/%Y").strftime("%Y-%m-%d")):
        sys.exit(f"{qdate} is not a working day")
    div = get_frame("Historical",f"""select Symbol,exDate,Amount as 'Div' from dbo.divs where convert(datetime,exDate,103) > convert(datetime,'{qdate}',103) order by convert(datetime,exDate,103)""")
    lastdiv = div["exDate"].iloc[1] if len(div) > 1 else '01/01/2100'
    df = get_frame("Historical",query_sql(qdate,lastdiv)).dropna()
    if qtime != None: df = df[df["Time"]==qtime]
    df = days(df,"Expiry","Date","Tenor")
    df = df.set_index("Symbol").join(div[div.index==0].set_index("Symbol")).reset_index()
    df = days(df,"exDate","Date","CumDivDays")
    df["Div"] = np.where(df["CumDivDays"]>df["Tenor"],0,df["Div"])
    df["CumDivDays"] = np.where(df["CumDivDays"]>df["Tenor"],0,df["CumDivDays"])
    return df[df["Expiry"]!=qdate].sort_values(by=["Tenor","Time","Strike"])

#Add rates interpolation option in main call
def rates(df,date,tenor):
    rate = get_frame("Historical",f"select * from dbo.rates where Date = '{date}'")    
    tenor_dict = {"12M":365,"6M":365/2,"3M":365/4,"1M":365/12,"1W":365/52.1429,"ON":1}
    rate["Tenor"] = rate["Tenor"].replace(tenor_dict)
    rate = rate.sort_values("Tenor")
    f = PchipInterpolator(rate["Tenor"],rate["Rate"])
    df["Rate"] = f(tenor)
    return df

def list2str(df):
    for x,y in zip(df.iloc[0],df.columns):
        if str(type(x)) == "<class 'list'>" or str(type(x)) == "<class 'numpy.ndarray'>":
            df[y] = df[y].apply(lambda z:",".join(map(str,z)))
    return df

def post_frame(df,database,table_name,ifexists): 
    df = list2str(df)       
    server="LAPTOP-206OR7PL\\SQLEXPRESS"
    database=database
    engine = sqlalchemy.create_engine("mssql+pyodbc://@"+server+"/"+
    database+"?trusted_connection=yes&driver=ODBC+Driver+13+for+SQL+Server")
    df.to_sql(name=table_name,con=engine,if_exists=ifexists,index=False)
#------------------------------------------------------------------------------------------------
#qdate,qtime = "15/05/2020","15:30" #ex 29/06/2020
#df = query(qdate,qtime)

    
    
    
    
    
