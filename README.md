# Bitcoin Price Prediction
### Predict bitcoin price using gold and S&amp;P 500 data implementing LSTM, Gradient Boosting Regression, and Random Forest


### data_engineering.py
Includes functions for downloading and modelling the price of BTC in USD

### Source Data:
* [API: cryptocompare.com for BTC and associated metrics](https://min-api.cryptocompare.com)
* [SPDR GoldShares (GLD) to represent gold prices](http://www.spdrgoldshares.com/usa/historical-data/)
* [Business Insider: SPDR Gold Shares for latest price](http://markets.businessinsider.com/etfs/spdr-gold-shares)
* [S&P Dow Jones Indices LLC, S&P 500, retrieved from FRED, Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/series/SP500)

### References:

* [SPDR Gold Shares Fact Sheet](http://www.spdrgoldshares.com/media/GLD/file/ETF-GLD_20171231.pdf)
* [Selecting Features for RandomForrest, datadive.net](http://blog.datadive.net/selecting-good-features-part-iii-random-forests/)
* [LMBA Gold Price](http://www.lbma.org.uk/lbma-gold-price)
* [Diving Into Data: Selecting Good Features](http://blog.datadive.net/selecting-good-features-part-iii-random-forests/)
* [Colah's Blog: Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [Machine Learning Mastery: Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
* [David Sheehan Blog: Predicting Cryptocurrency Prices With Deep Learning](https://dashee87.github.io/deep%20learning/python/predicting-cryptocurrency-prices-with-deep-learning/)
* [sklearn.ensemble](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
* [Wikipedia: SPDR ETFs](https://en.wikipedia.org/wiki/SPDR)
* [Wikipedia: Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
* [Wikipedia: Random Forest](https://en.wikipedia.org/wiki/Random_forest)
* [Wikipedia: Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning)
* [Wikipedia: Long short-term memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory)
