This is the implementation of the master thesis Some Aspects of Deep Portfolio Optimization. 

Zhengyao Jiang, Dixing Xu and Jinjun Liang puplished at the end of June 2017 A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem [arXiv:1706.10059] in which they presented a full exploitation deterministic deep on-policy gradient reinforcement learning system. The reinforcement learning agent of this system selects and manages a portfolio of various cryptocurrencies. In this thesis we analyze the impact of different input features to the deep neural network, a method to pretrain the deep neural network and different data normalization schemes on the policy function of the reinforcement learning agent. The analysis is done by considering the accumulative portfolio value over time, the sharpe ratio, the maximum drawdown and the portfolio weights. The different reinforcement learning systems are tested on two different data sets. The first data set represents a bull market and the second data set is a bearish market containing a market crash. The results of the analysis show that the mentioned modifications to the reinforcement learning system cause a significant change in the investment philosophy of the artificial portfolio manager.


## Platform Support
Python 3.5+ in windows and Python 2.7+/3.5+ in linux are supported.

## Dependencies
Install Dependencies via `pip install -r requirements.txt`

* tensorflow (>= 1.0.0)
* tflearn
* pandas
* ...

## User Guide

Check [User Guide](user_guide.md)
