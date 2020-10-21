# Vol-surface-parametrisation

![Example](https://github.com/sle14/Vol-surface-parametrisation/blob/master/examples/1.PNG?raw=true)

Minimisation

<img src="https://render.githubusercontent.com/render/math?math=k = K/F_{T}">
<img src="https://render.githubusercontent.com/render/math?math=w(k) = \frac{\theta_{t}}{2}(1 %2B k\rho(\theta_{t})\phi(\theta_{t}) %2B \sqrt{(k\phi(\theta_{t}) %2B \rho(\theta_{t}))^2 %2B (1-\rho(\theta_{t})^2)})">
<img src="https://render.githubusercontent.com/render/math?math=\rho(\theta_{t}) = ae^{-b\theta_{t}} %2B c">
<img src="https://render.githubusercontent.com/render/math?math=\phi(\theta_{t}) = \eta/\theta_{t}^{\gamma}(1 %2B \theta_{t})^{1-\gamma}">
<img src="https://render.githubusercontent.com/render/math?math=\sigma_{ssvi} = \sqrt{w(k)/t}">
<img src="https://render.githubusercontent.com/render/math?math=\epsilon = (\sigma_{ssvi} - \sigma_{quotes})^2">

Density

<img src="https://render.githubusercontent.com/render/math?math=g(k) = (1-\frac{kw'(k)}{2w(k)})^2-\frac{w'(k)^2}{4}(\frac{1}{w(k)} %2B \frac{1}{4}) %2B \frac{w''(k)}{2}">
<img src="https://render.githubusercontent.com/render/math?math=p(k) = \frac{g(k)}{\sqrt{2\pi w(k)}}e^{-\frac{d_{-}(k)}{2}^2}">
<img src="https://render.githubusercontent.com/render/math?math=d_{-}(k) = -\frac{k}{\sqrt{w}}\frac{\sqrt{w}}{2}">

Jumpwings

<img src="https://render.githubusercontent.com/render/math?math=v_{t} = \frac{\theta_{t}}{t}">
<img src="https://render.githubusercontent.com/render/math?math=\psi_{t} = \frac{1}{2}\rho(\theta_{t}) \phi(\theta_{t}) \sqrt{\theta}">
<img src="https://render.githubusercontent.com/render/math?math=p_{t} = \frac{1}{2}\sqrt{\theta_{t}}\phi(\theta_{t})(1-\rho)">
<img src="https://render.githubusercontent.com/render/math?math=c_{t} = \frac{1}{2}\sqrt{\theta_{t}}\phi(\theta_{t})(1 %2B \rho)">
<img src="https://render.githubusercontent.com/render/math?math=\widetilde{v_{t}} = \frac{\theta_{t}}{t}(1-\rho(\theta_{t})^2)">

European options value

<img src="https://render.githubusercontent.com/render/math?math=c(S,T) = Se^{-qT}N(d1(S,K,T))-Ke^{-rT}N(d2(S,K,T))">
<img src="https://render.githubusercontent.com/render/math?math=p(S,T) = Ke^{-rT}N(-d2(S,K,T))-Se^{-qT}N(-d1(S,K,T))">

Early exercise premium

<img src="https://render.githubusercontent.com/render/math?math=EEP_{Call}(S,T) = \int_{0}^{T} [qB_{t}e^{-q(T-t)}N(d1(S,B_{t},T-t))-rKe^{-r(T-t)}N(d2(S,B_{t},T-t))] dt">
<img src="https://render.githubusercontent.com/render/math?math=EEP_{Put}(S,T) = \int_{0}^{T} [rKe^{-r(T-t)}N(-d2(S,B_{t},T-t))-qB_{t}e^{-q(T-t)}N(-d1(S,B_{t},T-t))] dt">

American options value

<img src="https://render.githubusercontent.com/render/math?math=C(S,T) = c(S,T) %2B EEP_{Call}(S,T)">
<img src="https://render.githubusercontent.com/render/math?math=P(S,T) = p(S,T) %2B EEP_{Put}(S,T)">

Spread function

<img src="https://render.githubusercontent.com/render/math?math=H_{Call}(S) = H_{0} %2B H_{1}max(S-K_{h},0)">
<img src="https://render.githubusercontent.com/render/math?math=H_{Put}(S) = H_{0} %2B H_{1}max(K_{h}-S,0)">

Terminal condition

<img src="https://render.githubusercontent.com/render/math?math=B_{T,Call} = Kmax(1,\frac{r}{q})">
<img src="https://render.githubusercontent.com/render/math?math=B_{T,Put} = Kmin(1,\frac{r}{q})">

Boundary conditions: high-contact and value-matching

<img src="https://render.githubusercontent.com/render/math?math=\lim_{S\rightarrow B_{t}} \frac{\partial P(S,K,t)}{\partial S} = -1">
<img src="https://render.githubusercontent.com/render/math?math=\lim_{S\rightarrow B_{t}} \frac{\partial C(S,K,t)}{\partial S} = 1">

<img src="https://render.githubusercontent.com/render/math?math=\lim_{S\rightarrow B_{t}} P(S,K,t) = K - B_{t}">
<img src="https://render.githubusercontent.com/render/math?math=\lim_{S\rightarrow B_{t}} C(S,K,t) = B_{t} - K">

Boundary solution

<img src="https://render.githubusercontent.com/render/math?math=B_{t} - K = c(B_{t},T-t) %2B EEP_{Call}(B_{t},T-t) - H(B_{t})">
<img src="https://render.githubusercontent.com/render/math?math=K - B_{t} = p(B_{t},T-t) %2B EEP_{Put}(B_{t},T-t) - H(B_{t})">

Trapezoid rule

<img src="https://render.githubusercontent.com/render/math?math=\int_{a}^{b}f(x)dx \approx \Delta x[\frac{(f(x_{a}) %2B f(x_{b}))}{2} %2B \sum_{n=1}^{N-1}f(x_{n})]">

# References
[1] Gatheral, J., Jacquier, A., Arbitrage-Free SVI Volatility Surfaces. Quantitative Finance, Vol. 14, No. 1, 59-71, 2014, http://dx.doi.org/10.2139/ssrn.2033323

[2] Kallast, S., Kivinukk,A. Pricing and Hedging American Options Using Approximations by Kim Integral Equations. Review of Finance, Volume 7, Issue 3, 2003, Pages 361–383, https://doi.org/10.1023/B:EUFI.0000022128.44728.4c

[3] Figlewski, S., An American Call IS Worth More than a European Call: The Value of American Exercise When the Market is Not Perfectly Liquid. New York University - Stern School of Business, 2019, http://dx.doi.org/10.2139/ssrn.2977494
