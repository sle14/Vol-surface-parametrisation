# Vol Surface Parametrisation

#Add voila link here...

## Contents

- [About](#about)
- [Features](#features)
- [Guide](#guide)
- [Formulas](#formulas)
- [References](#references)

## About

My initial goal was to find robust way of generating arb-free vol surface on single stock option quotes, so that I would be able to determine change in portfolio liquidation value for change in moments of vol, spreads, cost of carry, price of underlying and time. I have decided to use Gatheral's Surface Stohastic Volatility Inspired (SSVI) method for this, but had few issues with poor fit, which was expected as I was using vols from american quotes which explode in wings, particularly before the ex-date. So instead of direct minimisation on raw vols I had to first de-americanise them, by deducting the premium they command with respect to european quotes.

For this I went along with Kim integral approximation method instead of trees, as it has a very useful feature of decomposition of american options value into european base and early exercise values. However, as this method involves numerical solution of quadrature formulas on each step exercise boundary, even with numpy vectorisation and broadcasting it was still a bit too slow. So I wrote the base class in C++ and used ctypes lib to call its functions from Python. Having derived de-americanised vols, I was able to fit SSVI on them directly.

Calibration to the market is done using SLSQP algorithm with constrains and bounds to prevent arbitrage in the surface. Risk neutral density is derived using fitted SSVI parameters with explicit differentiation of BSM formula and primes of surface function. I have changed Jump-wing parameters from 5 to 3 where we now have ATMF Vol, 1st and 2nd derivatives of ATMF variance (skew & kurtosis), when we shock these parameters we can invert them back to raw parameters to get the new vol surface.

All the above is packed into numpy structs to allow for better handling of multi-dimensionality and memory optimisation, ie in case of plotting payoffs and liquidation values wrt dSpot and dTime. I have used plotly and ipwidgets for interface, and voila for the server. 

## Features

- Breakdown of positions liquidation value by Greeks, Spread and EEP for given changes in 11 factors by expiry group.
- Plotting of payoffs and values for dS and dT ranges that use the factor changes as specified by user and consider Spread and EEP.
- Density and total variance plots for referring to absence of arbitrage within the newly generated surface specified by user.
- Daily data with 10 minute interval for 40 listed US single stock option names, with 3 front month expiries per each where available.
- Ability to specify weights for the SSVI fit residual based on gaussian density location and scale parameters.

## Guide

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/PANEL_1.PNG?raw=true)

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/PANEL_2.PNG?raw=true)

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/FIG_LHS_VOL_1.PNG?raw=true)

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/FIG_LHS_VOL_2.PNG?raw=true)

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/FIG_LHS_VOL_3.PNG?raw=true)

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/TABLE_1.PNG?raw=true)

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/FIG_PAY_1.PNG?raw=true)

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/FIG_PAY_2.PNG?raw=true)

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/FIG_RHS_RND_1.PNG?raw=true)

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/FIG_LHS_CHS_1.PNG?raw=true)

## Formulas

**Vol Calibration**

SSVI fits on total ATMF variance (θ) and logstrike (k) space, where we can specify a type of convexity function (φ) and spot-vol correlation function (ρ):

<img src="https://render.githubusercontent.com/render/math?math=w(k) = \frac{\theta_{t}}{2}(1 %2B k\rho(\theta_{t})\phi(\theta_{t}) %2B \sqrt{(k\phi(\theta_{t}) %2B \rho(\theta_{t}))^2 %2B (1-\rho(\theta_{t})^2)})">
<img src="https://render.githubusercontent.com/render/math?math=k = K/F_{T}">
<img src="https://render.githubusercontent.com/render/math?math=\rho(\theta_{t}) = ae^{-b\theta_{t}} %2B c">
<img src="https://render.githubusercontent.com/render/math?math=\phi(\theta_{t}) = \eta\theta_{t}^{-\lambda}">

Then we need to minimise the below residual:

<img src="https://render.githubusercontent.com/render/math?math=\sigma_{ssvi} = \sqrt{w(k,\theta_{t},\phi,\rho)/t}">
<img src="https://render.githubusercontent.com/render/math?math=\epsilon = arg min(\sigma_{ssvi} - \sigma_{quotes})^2">

 
**Risk Neutral Density**

Explicit differentiation of BSM formula leads to:

<img src="https://render.githubusercontent.com/render/math?math=g(k) = (1-\frac{kw'(k)}{2w(k)})^2-\frac{w'(k)^2}{4}(\frac{1}{w(k)} %2B \frac{1}{4}) %2B \frac{w''(k)}{2}">

Where we need to find 1st and 2nd derivatives of total variance wrt logstrike, so we just take these from the surface function w(k):

<img src="https://render.githubusercontent.com/render/math?math=w(k) = \frac{\theta}{2}[1 %2B k\rho\phi %2B \sqrt{(k\phi %2B \rho)^{2} + (1 - \rho^{2})}]">
<img src="https://render.githubusercontent.com/render/math?math=w(k){}' = \frac{\theta}{2}[\frac{k\phi^2 %2B \rho\phi}{\sqrt{(k\phi %2B \rho)^2 - \rho^2 %2B 1} %2B \rho\phi}]">
<img src="https://render.githubusercontent.com/render/math?math=w(k){}'' = -[ \frac{\theta\phi^2(\rho^2-1)}{2((k\phi %2B \rho)^2-\rho^2 %2B 1)^\frac{3}{2}} ]">

And to get probability density:

<img src="https://render.githubusercontent.com/render/math?math=d_{-}(k) = -\frac{k}{\sqrt{w}}\frac{\sqrt{w}}{2}">
<img src="https://render.githubusercontent.com/render/math?math=p(k) = \frac{g(k)}{\sqrt{2\pi w(k)}}e^{-\frac{d_{-}(k)}{2}^2}">

 
**Jump-wings**

Jump-wings I have used are a bit different from the ones proposed by Gatheral, but the principle and the goal are the same. We want to know how will vol surface behave if we change 3 vol factors: level (σ), skew (ψ) and kurtosis (κ). 

Converting from raw to jw:

<img src="https://render.githubusercontent.com/render/math?math=\sigma_{t} = \sqrt{\theta/t}">
<img src="https://render.githubusercontent.com/render/math?math=\psi_{t} = [ \frac{1}{2}\rho\phi\sqrt{\theta}]/t">
<img src="https://render.githubusercontent.com/render/math?math=\kappa_{t} = [ \frac{1}{2}\phi^2\theta(1 - \rho^2))]/t">

Converting from jw to raw:

<img src="https://render.githubusercontent.com/render/math?math=\theta_{t} = \sigma^2t">
<img src="https://render.githubusercontent.com/render/math?math=\rho = \frac{\psi t}{\sqrt{\frac{\kappa t}{2} + (\psi t)^2}}">
<img src="https://render.githubusercontent.com/render/math?math=\phi= \frac{2\psi t}{\rho\sqrt{\theta}}">

 
**Spread function**

We fit the the below functions on the raw spreads to get the three parameters:

<img src="https://render.githubusercontent.com/render/math?math=H_{Call}(S) = H_{0} %2B H_{1}max(S-K_{h},0)">
<img src="https://render.githubusercontent.com/render/math?math=H_{Put}(S) = H_{0} %2B H_{1}max(K_{h}-S,0)">

Spread widens for deeper ITM options, we make no assumption here as to where it starts to widen, but Kh should be around ATM/F.

**H0:** minimum spread
**H1:** slope the spread climb
**Kh:** strike of spread climb

 
**American options value**

Is composed of european base value and early exercise premium (EEP):

<img src="https://render.githubusercontent.com/render/math?math=C(S,T) = c(S,T) %2B EEP_{Call}(S,T)">
<img src="https://render.githubusercontent.com/render/math?math=P(S,T) = p(S,T) %2B EEP_{Put}(S,T)">

European options value:

<img src="https://render.githubusercontent.com/render/math?math=c(S,T) = Se^{-qT}N(d1(S,K,T))-Ke^{-rT}N(d2(S,K,T))">
<img src="https://render.githubusercontent.com/render/math?math=p(S,T) = Ke^{-rT}N(-d2(S,K,T))-Se^{-qT}N(-d1(S,K,T))">

Where EEP is the integral of the boundary price (B) from present till the expiry (T):

<img src="https://render.githubusercontent.com/render/math?math=EEP_{Call}(S,T) = \int_{0}^{T} [qB_{t}e^{-q(T-t)}N(d1(S,B_{t},T-t))-rKe^{-r(T-t)}N(d2(S,B_{t},T-t))] dt">
<img src="https://render.githubusercontent.com/render/math?math=EEP_{Put}(S,T) = \int_{0}^{T} [rKe^{-r(T-t)}N(-d2(S,B_{t},T-t))-qB_{t}e^{-q(T-t)}N(-d1(S,B_{t},T-t))] dt">

 
**Boundary conditions**

Boundary price is the price at which returns from selling options and execising it are the same, and is subject to below conditions.

Terminal condition:

<img src="https://render.githubusercontent.com/render/math?math=B_{T,Call} = Kmax(1,\frac{r}{q})">
<img src="https://render.githubusercontent.com/render/math?math=B_{T,Put} = Kmin(1,\frac{r}{q})">

High-contact condition:

<img src="https://render.githubusercontent.com/render/math?math=\lim_{S\rightarrow B_{t}} \frac{\partial P(S,K,t)}{\partial S} = -1">
<img src="https://render.githubusercontent.com/render/math?math=\lim_{S\rightarrow B_{t}} \frac{\partial C(S,K,t)}{\partial S} = 1">

Value-matching condition:

<img src="https://render.githubusercontent.com/render/math?math=\lim_{S\rightarrow B_{t}} P(S,K,t) = K - B_{t}">
<img src="https://render.githubusercontent.com/render/math?math=\lim_{S\rightarrow B_{t}} C(S,K,t) = B_{t} - K">

 
**Boundary solution**

We derive boundary via backward induction where we start at expiry and move through backwards in time. As per Terminal condition we can already find out what is the boundary on the expiry date. 

<img src="https://render.githubusercontent.com/render/math?math=B_{t} - K = c(B_{t},T-t) %2B EEP_{Call}(B_{t},T-t) - H(B_{t})">
<img src="https://render.githubusercontent.com/render/math?math=K - B_{t} = p(B_{t},T-t) %2B EEP_{Put}(B_{t},T-t) - H(B_{t})">

Note that we deduct half-spread from the above to adjust the difference in liquidity between the spot and option, in addition to the fact that spread widens as options gets deeper ITM which is exactly where the boundary is located.

We use Trapezoid rule to find the quadrature for the EEP component and final value as we integrate all the boundaries till the expiry:

<img src="https://render.githubusercontent.com/render/math?math=\int_{a}^{b}f(x)dx \approx \Delta x[\frac{(f(x_{a}) %2B f(x_{b}))}{2} %2B \sum_{n=1}^{N-1}f(x_{n})]">

 
## References
[1] Gatheral, J., Jacquier, A., Arbitrage-Free SVI Volatility Surfaces. Quantitative Finance, Vol. 14, No. 1, 59-71, 2014, http://dx.doi.org/10.2139/ssrn.2033323

[2] Kallast, S., Kivinukk,A. Pricing and Hedging American Options Using Approximations by Kim Integral Equations. Review of Finance, Volume 7, Issue 3, 2003, Pages 361–383, https://doi.org/10.1023/B:EUFI.0000022128.44728.4c

[3] Figlewski, S., An American Call IS Worth More than a European Call: The Value of American Exercise When the Market is Not Perfectly Liquid. New York University - Stern School of Business, 2019, http://dx.doi.org/10.2139/ssrn.2977494
