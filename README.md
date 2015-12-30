# time_series

This is a playground to understand time series and implement related algorithms. 
Most implemented algorithms are taken from Time Series: Theory and Methods by Brockwell and Davis and mathematical notation is close. One notebook contains a quick summary of a few important points from some chapters and some solutions to the exercises. The other notebook contains examples of some implemented algorithms, these include:

* Yule Walker equations to fit AR models 
* Durbin Levinson recursions, maximal likelihood (both via innovation algorithm and Kalman recursions) and least squaress to fit ma and arma models
* various tests for model order selection (AICC, BIC, FPE)
* goodnes of fit tests (turning point test)
* impulse response function
* transformations (mean adjustment, box-cox)
* theoretical and sample acf and pacf

Most functions are unittested.
