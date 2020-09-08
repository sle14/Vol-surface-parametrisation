# Vol-surface-parametrisation
Parametrisation of vol surface using Gatheral's SVI methodology

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

Moments


# References
[1] Gatheral, J., Jacquier, A., Arbitrage-Free SVI Volatility Surfaces. Quantitative Finance, Vol. 14, No. 1, 59-71, 2014, http://dx.doi.org/10.2139/ssrn.2033323
