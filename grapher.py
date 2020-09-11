from scipy.interpolate import InterpolatedUnivariateSpline as spline
from plotly.subplots import make_subplots
template,transparent = "plotly_dark","rgba(0,0,0,0)"
import plotly.graph_objects as go
import numpy as np
import optimiser

def payoff_arrays(R,S,K,T):
    x = [min(K),S,max(K)]
    y = [T]*3
    z = [max(0,S-min(K)),0,max(0,S-max(K))] if R=="Call" else [max(0,min(K)-S),0,max(0,max(K)-S)]
    return x,y,z

def interp_z(a,x,y):
    f = spline(x,y,k=2)
    return f(a)

def plot2d_layout_subplot(fig,x1_title,x2_title,y1_title,y2_title):
    fig.update_layout(template=template,paper_bgcolor=transparent,plot_bgcolor=transparent,height=600,width=1500)
    fig.update_xaxes(title_text=x1_title,row=1,col=1)
    fig.update_yaxes(title_text=y1_title,row=1,col=1)
    fig.update_xaxes(title_text=x2_title,row=1,col=2)
    fig.update_yaxes(title_text=y2_title,row=1,col=2)
    return fig
    
def plot2d_layout(fig,x_title,y_title):
    fig.update_layout(template=template,paper_bgcolor=transparent,plot_bgcolor=transparent,autosize=False,width=900,height=800)
    fig.update_layout(xaxis_title=x_title,yaxis_title=y_title)
    return fig

def plot3d_layout_subplot(fig,x_title,y_title,z1_title,z2_title):
    fig.update_layout(template=template,paper_bgcolor=transparent,plot_bgcolor=transparent,autosize=False,width=1500,height=700)
    fig.update_layout(scene=dict(xaxis=dict(backgroundcolor=transparent,),xaxis_title=x_title,
                                 yaxis=dict(backgroundcolor=transparent,),yaxis_title=y_title,
                                 zaxis=dict(backgroundcolor=transparent,),zaxis_title=z1_title),
                      scene2=dict(xaxis=dict(backgroundcolor=transparent,),xaxis_title=x_title,
                                 yaxis=dict(backgroundcolor=transparent,),yaxis_title=y_title,
                                 zaxis=dict(backgroundcolor=transparent,),zaxis_title=z2_title),)
    return fig

def plot3d_layout(fig,x_title,y_title,z_title):
    fig.update_layout(template=template,paper_bgcolor=transparent,plot_bgcolor=transparent,autosize=False,width=900,height=800)              
    fig.update_layout(scene=dict(xaxis=dict(backgroundcolor=transparent,),xaxis_title=x_title,
                                 yaxis=dict(backgroundcolor=transparent,),yaxis_title=y_title,
                                 zaxis=dict(backgroundcolor=transparent,),zaxis_title=z_title),)
    return fig

def plot3d_boundaries(df,num):
    tenors = sorted(df["Tenor"].unique())
    dfc = df[df["Tenor"]==tenors[num]]
    fig = make_subplots(rows=1,cols=2,specs=[[{'type':'scatter3d'},{'type':'scatter3d'}]])
    for K in dfc["Strike"].unique():
        boundaries = dfc[dfc["Strike"]==K]["eebC"].iloc[0].split(",")
        z = [float(i) for i in boundaries]
        dt = tenors[num]/len(boundaries)
        y = np.flip(np.array([i*dt for i in range(len(boundaries))]))
        x = [K]*len(boundaries)
        trace_3d(fig,x,y,z,f"EEB Call {K}","lines","gray",idx=[1,1])
    for K in dfc["Strike"].unique():
        boundaries = dfc[dfc["Strike"]==K]["eebP"].iloc[0].split(",")
        z = [float(i) for i in boundaries]
        dt = tenors[num]/len(boundaries)
        y = np.flip(np.array([i*dt for i in range(len(boundaries))]))
        x = [K]*len(boundaries)
        trace_3d(fig,x,y,z,f"EEB Put {K}","lines","gray",idx=[1,2])
    fig = plot3d_layout_subplot(fig,"Strike","Tenor","Spot","Spot")
    fig.show()

def plot3d_single(fig,df,S_col,x_col,z_col,atm_plots=False,div_plot=False,payoff_plot=False,params_plot=False,idx=None):
    if div_plot == True:
        div = max(df["Div"])
        T = max(df["CumDivDays"])
        if T != 0: 
            fig = trace_3d(fig,[min(df[x_col]),max(df[x_col])],[T]*2,[min(df[z_col])]*2,f"{T} Div {div}","lines","orange",idx=idx,group="Divs")
    for t in df["Tenor"].unique():
        dfx = df[df["Tenor"]==t]
        S,F,T,K = df[S_col].iloc[0],dfx["Forward"].iloc[0],dfx["Tenor"],dfx["Strike"]
        fig = trace_3d(fig,dfx[x_col],T,dfx[z_col],f"{t} Quoted {z_col}","markers","white",2,idx=idx,group="z_col") #Quoted px,vol,var,totvar
        if payoff_plot != False:
            R = "Call" if "AC" in z_col or "EC" in z_col else "Put"
            x,y,z = payoff_arrays(R,S,dfx["Strike"],t) 
            if x_col == "Moneyness":
                x = x/S
            elif x_col == "LogStrike":
                x = np.log(x/F)
            fig = trace_3d(fig,x,y,z,f"{t} {R} Payoff","lines","skyblue",idx=idx,group="Payoffs")
        if params_plot != False:
            k = np.log(K/F) if x_col != "LogStrike" else dfx["LogStrike"]
            params = [float(i) for i in dfx["SurfaceParams"].iloc[0].split(",")]
            w = optimiser.SSVI().surface(np.array(k),dfx["TotAtmVar"].iloc[0],*params)
            if "Vol" in z_col:
                w = np.sqrt(w/(t/365)) 
            elif "Var" in z_col and not "Tot" in z_col:
                w = w/(t/365)
            fig = trace_3d(fig,dfx[x_col],T,w,f"{t} Fitted {z_col}","lines","gray",idx=idx,group="Fitted") #Fitted
        if atm_plots != False:
            if x_col == "Moneyness":
                x0,x1 = 1,F/S
            elif x_col == "LogStrike":
                x0,x1 = np.log(S/F),0
            else:
                x0,x1 = S,F
            fig = trace_3d(fig,[x0],[t],[interp_z(S,K,dfx[z_col])],f"{t} ATM","markers","green",6,"cross",idx=idx,group="ATM") #ATM
            fig = trace_3d(fig,[x1],[t],[interp_z(F,K,dfx[z_col])],f"{t} ATMF","markers","yellow",6,"cross",idx=idx,group="ATMF") #ATMF
    return fig

#So add function on top of this that is used for subplot
def plot3d_quotes(df,S_col,x_col,z_col,atm_plots=False,div_plot=False,payoff_plot=False,params_plot=False,z_col2=None):
    if z_col2 != False:
        fig = make_subplots(rows=1,cols=2,specs=[[{'type':'scatter3d'},{'type':'scatter3d'}]])
        fig = plot3d_single(fig,df,S_col,x_col,z_col,atm_plots,div_plot,payoff_plot,params_plot,[1,1])
        fig = plot3d_single(fig,df,S_col,x_col,z_col2,atm_plots,div_plot,payoff_plot,params_plot,[1,2])
        fig = plot3d_layout_subplot(fig,x_col,"Tenor",z_col,z_col2)
    else:
        fig = go.Figure()
        fig = plot3d_single(fig,df,S_col,x_col,z_col,atm_plots,div_plot,payoff_plot,params_plot)
        fig = plot3d_layout(fig,x_col,"Tenor",z_col)
    fig.show()

def trace_2d(fig,x,y,name,idx=None):
    if idx != None:
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",name=name),row=idx[0],col=idx[1])
    else:
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",name=name))
    return fig

def trace_3d(fig,x,y,z,name,mode,colour,size=None,symbol=None,idx=None,group=None):
    size = 0 if size == None else size
    symbol = "circle" if symbol == None else symbol
    if idx != None:
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,legendgroup=group,mode=mode,marker=dict(size=size,color=colour,symbol=symbol),line=dict(color=colour,width=4),name=name),row=idx[0],col=idx[1])
    else:
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,legendgroup=group,mode=mode,marker=dict(size=size,color=colour,symbol=symbol),line=dict(color=colour,width=4),name=name))
    return fig
    
def plot2d_fitted(df,V,T,F,a,b,m):
    params = [float(i) for i in df["SurfaceParams"].iloc[0].split(",")]
    K = np.linspace(a,b,m)
    fig = make_subplots(rows=1,cols=2)
    for v,t,fwd in zip(df[V].unique(),df[T].unique(),df[F].unique()):
        k = np.log(K/fwd)
        w = optimiser.SSVI().surface(k,v,*params)
        fig = trace_2d(fig,k,w,f"{t} TotVar",[1,1])
    for v,t,fwd in zip(df[V].unique(),df[T].unique(),df[F].unique()):
        k = np.log(K/fwd)
        f = spline(k,optimiser.SSVI().rnd(k,v,*params),k=2)
        k = np.linspace(k[0],k[-1],m)
        P = [(k[1]-k[0])*p for p in f(k)]
        fig = trace_2d(fig,k,P,f"{t} Density area {round(sum(P),6)}",[1,2])
    fig = plot2d_layout_subplot(fig,"LogStrike","LogStrike","TotalVariance","Density")
    fig.show()