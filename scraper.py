from bs4 import BeautifulSoup
import pandas as pd
import requests
import query

def links(currency):
    if currency == "EUR":
        return {"12M":"https://www.global-rates.com/en/interest-rates/euribor/euribor-interest-12-months.aspx",
                "6M":"https://www.global-rates.com/en/interest-rates/euribor/euribor-interest-6-months.aspx",
                "3M":"https://www.global-rates.com/en/interest-rates/euribor/euribor-interest-3-months.aspx",
                "1M":"https://www.global-rates.com/en/interest-rates/euribor/euribor-interest-1-month.aspx",
                "1W":"https://www.global-rates.com/en/interest-rates/euribor/euribor-interest-1-week.aspx",
                "ON":"https://www.global-rates.com/en/interest-rates/eonia/eonia.aspx"}
    elif currency == "USD":
        return {"6M":"https://www.global-rates.com/en/interest-rates/libor/american-dollar/usd-libor-interest-rate-6-months.aspx"}

def parse_dates(df,cols:list,sort_col,dformat):
    for i in cols: 
        df[i] = pd.to_datetime(df[i],format=dformat)
        if i == sort_col: df = df.sort_values(by=[sort_col]) 
    return df

def rates(currency,tenor):
    if currency == "EUR":
        name = "EURIBOR" if tenor != "ON" else "EONIA"
    else:
        name = "LIBOR"
    url_dict = links(currency)
    url = url_dict[tenor]
    res = requests.get(url).text
    soup = BeautifulSoup(res,"html.parser")
    tables = soup.findAll("table")[6].findAll("tr")[14:26]
    df = pd.DataFrame(columns=["Date","Type","Tenor","Rate","Currency"])
    for caption in tables:
        caption_date = caption.text.strip().rsplit('0.',1)[0].replace("-","")
        date = pd.to_datetime(caption_date)
        rate = round(float(caption.text.strip()[-8:][:6])/100,5)
        row = [date,name,tenor,rate,currency]
        print(row)
        df.loc[len(df)] = row
    df = pd.concat([df,query.get("Static","select * from dbo.rates")],axis=0)
    df = df.drop_duplicates(["Date","Tenor"])
    df = parse_dates(df,["Date"],"Date","%d/%m/%Y")
    query.post(df,"Static","rates","replace")
    print("Posted")
    
rates("USD","6M")







