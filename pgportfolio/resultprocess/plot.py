from __future__ import absolute_import, print_function, division
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Polygon
from pgportfolio.tools.data import panel_fillna
import sqlite3
from matplotlib import rc
import pandas as pd
import logging
import json
import numpy as np
import time
import datetime
from pgportfolio.tools.trade import get_coin_name_list as get_coin_name_list
#import of measures
from pgportfolio.tools.indicator import max_drawdown, sharpe, positive_count, negative_count, moving_accumulate
from pgportfolio.tools.configprocess import parse_time, check_input_same
#backtest import
from pgportfolio.tools.shortcut import execute_backtest

# the dictionary of name of indicators mapping to the function of related indicators
# input is portfolio changes
INDICATORS = {"portfolio value": np.prod,
              "sharpe ratio": sharpe,
              "max drawdown": max_drawdown,
              "positive periods": positive_count,
              "negative periods": negative_count,
              "postive day": lambda pcs: positive_count(moving_accumulate(pcs, 48)),
              "negative day": lambda pcs: negative_count(moving_accumulate(pcs, 48)),
              "postive week": lambda pcs: positive_count(moving_accumulate(pcs, 336)),
              "negative week": lambda pcs: negative_count(moving_accumulate(pcs, 336)),
              "average": np.mean}

NAMES = {"best": "Best Stock (Benchmark)",
         "crp": "UCRP (Benchmark)",
         "ubah": "UBAH (Benchmark)",
         "anticor": "ANTICOR",
         "olmar": "OLMAR",
         "pamr": "PAMR",
         "cwmr": "CWMR",
         "rmr": "RMR",
         "ons": "ONS",
         "up": "UP",
         "eg": "EG",
         "bk": "BK",
         "corn": "CORN",
         "m0": "M0",
         "wmamr": "WMAMR"
         }
#used by main for ploting portfolio value - time
def plot_backtest(config, algos, labels=None, datess=None, coinlist=None):
    """
    @:param config: config dictionary
    @:param algos: list of strings representing the name of algorithms or index of pgportfolio result
    """
    results = []
    #goes through all the named algos
    for i, algo in enumerate(algos):
        if algo.isdigit():
            #appends from summary, method in plot.py, appends the portfolio value to results
            results.append(np.cumprod(_load_from_summary(algo, config, "result")))
            logging.info("load index "+algo+" from csv file")
        else:
            logging.info("start executing "+algo)
            results.append(np.cumprod(execute_backtest(algo, config)))
            logging.info("finish executing "+algo)

    start, end = _extract_test(config)

    #returns even timestamp between start and end
    timestamps = np.linspace(start, end, len(results[0]))
    dates = [datetime.datetime.fromtimestamp(int(ts)-int(ts)%config["input"]["global_period"])
             for ts in timestamps]

    weeks = mdates.WeekdayLocator()
    days = mdates.DayLocator()

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"],
                  "size": 8})

    """
    styles = [("-", None), ("--", None), ("", "+"), (":", None),
              ("", "o"), ("", "v"), ("", "*")]
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5)
    for i, pvs in enumerate(results):
        if len(labels) > i:
            label = labels[i]
        else:
            label = NAMES[algos[i]]
        ax.semilogy(dates, pvs, linewidth=1, label=label)
        #ax.plot(dates, pvs, linewidth=1, label=label)

    plt.ylabel("portfolio value $p_t/p_0$", fontsize=12)
    plt.xlabel("time", fontsize=12)
    xfmt = mdates.DateFormatter("%m-%d %H:%M")
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_minor_locator(days)
    datemin = dates[0]
    datemax = dates[-1]
    ax.set_xlim(datemin, datemax)

    ax.xaxis.set_major_formatter(xfmt)
    plt.grid(True)
    plt.tight_layout()
    #plt.axhline(y=1.28293, color='r', linestyle='--', label='y = 1.28293')
    #plt.axhline(y=2.12875, color='r', linestyle='--', label='y = 2.12875')
    #plt.axhline(y=1.15624, color='r', linestyle='--', label='y = 1.15624')
    #plt.axhline(y=1.71, color='r', linestyle='--', label='y = 1.71')
    ax.legend(loc="upper left", prop={"size": 10})
    fig.autofmt_xdate()

    plt.savefig("result.eps", bbox_inches='tight',
                pad_inches=0)
    plt.show()
    plt.close()

    for i, algo in enumerate(algos):
        weights = _load_from_summary(algo, config, "weight")
        weights = np.cumsum(weights, 1)

        if datess is not None:
            startdate = datetime.datetime.strptime(datess[0], '%Y %m %d %H %M')
            enddate = datetime.datetime.strptime(datess[1], '%Y %m %d %H %M')
            startpoint = dates.index(startdate)
            endpoint = dates.index(enddate)
        else:
            startpoint = 0
            endpoint = len(timestamps)-2
        fig = plt.figure(labels[i])

        ######
        """
        startdate_in_sec = int(timestamps[startpoint]) - int(timestamps[startpoint]) % config["input"][
            "global_period"]
        enddate_in_sec = int(timestamps[endpoint]) - int(timestamps[startpoint]) % config["input"][
            "global_period"]
        for coin in coinlist:
            panel = get_panel(start=startdate_in_sec, end=enddate_in_sec, period=1800, coinlist=[coin])
            coinchart = panel['SUM(volume)'].tolist()

            fig = plt.figure(coin)
            ax1 = fig.add_subplot(111)
            print(len(dates[startpoint:endpoint+1]))
            print(len(coinchart[:-32]))
            ax1.plot(dates[startpoint:endpoint + 1], coinchart[:-32], label=coin)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
                       fancybox=True, shadow=True, ncol=1)
            plt.show()
            plt.close()

       ###
        """




        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        if coinlist is not None:
            ax3 = ax1.twiny()
            ax4 = ax1.twiny()
            ax5 = ax1.twiny()
        number_of_coins = config["input"]["coin_number"]+1
        xy = []
        labelss = get_coin_name_list(config=config, online=False)
        labelss.insert(0, 'BTC')
        colors = ["#7e1e9c", "#15b01a", "#0343df", "#ff81c0", "#06470c", "#e50000", "#95d0fc", "#029386", "#f97306",
                  "#96f97b", "#c20078", "#ffff14"]
        #prepare polygon points
        for j in range(number_of_coins):
            xy.append([timestamps[startpoint], 0])
            for y in range(startpoint,endpoint+1):
                xy.append([timestamps[y], weights[y][11-j]])
            xy.append([timestamps[endpoint+1], 0])
            xy = np.asarray(xy)
            polygon = Polygon(xy, color=colors[11-j], closed=True, label=labelss[11-j])
            ax1.add_patch(polygon)
            xy = []
        ax1.set_xlabel(r"Timestamps")
        plt.ylabel("portfolio weights", fontsize=12)

        datemin = timestamps[startpoint]
        datemax = timestamps[endpoint]
        ax1.set_xlim(datemin, datemax)
        ax2.plot(dates[startpoint:endpoint+1], np.ones(len(dates[startpoint:endpoint+1])))
        ax2.cla()
        xfmt = mdates.DateFormatter("%m-%d %H:%M")
        ax2.xaxis.set_major_locator(weeks)
        ax2.xaxis.set_minor_locator(days)
        datemin = dates[startpoint]
        datemax = dates[endpoint]
        ax2.set_xlim(datemin, datemax)
        ax2.set_ylim(0, 1)
        ax1.set_ylim(0, 1)
        ax2.xaxis.set_major_formatter(xfmt)
        ax2.set_xlabel(r"Dates")
        if coinlist is not None:
            startdate_in_sec = int(timestamps[startpoint]) - int(timestamps[startpoint]) % config["input"][
                "global_period"]
            enddate_in_sec = int(timestamps[endpoint]) - int(timestamps[startpoint]) % config["input"][
                "global_period"]
            coinchart=[]
            for coin in coinlist:
                panel = get_panel(start=startdate_in_sec, end=enddate_in_sec, period=1800, coinlist=[coin])
                coinchart.append(panel['close'].tolist())
            coinchart[2] = [1 / x for x in coinchart[2]]
            for maxim in range(3):
                maximum = max(coinchart[maxim])
                coinchart[maxim] = [x/maximum for x in coinchart[maxim]]
            ax3.plot(dates[startpoint:endpoint + 1], coinchart[0][:-3], label=coinlist[0], color='pink', linewidth=3.0)

            ax4.plot(dates[startpoint:endpoint + 1], coinchart[1][:-3], label=coinlist[1], color='black', linewidth=3.0)

            ax5.plot(dates[startpoint:endpoint + 1], coinchart[2][:-3], label=coinlist[2], color='navy', linewidth=3.0)

            ax3.legend(loc='upper center', bbox_to_anchor=(0, 0.05),
                       fancybox=True, shadow=True)
            ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05),
                       fancybox=True, shadow=True)
            ax5.legend(loc='upper center', bbox_to_anchor=(1, 0.05),
                       fancybox=True, shadow=True)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=number_of_coins)
        plt.show()
        plt.close()
    """
    startdate_in_sec = int(timestamps[startpoint]) - int(timestamps[startpoint]) % config["input"][
            "global_period"]
    enddate_in_sec = int(timestamps[endpoint]) - int(timestamps[startpoint]) % config["input"][
            "global_period"]
    for coin in coinlist:
        panel = get_panel(start=startdate_in_sec, end=enddate_in_sec, period=1800, coinlist=[coin])
        coinchart = panel['close'].tolist()
        fig = plt.figure(coin)
        ax1 = fig.add_subplot(111)
        ax1.plot(dates[startpoint:endpoint + 1], coinchart[:-3], label=coin)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
        fancybox = True, shadow = True, ncol = 1)
        plt.show()
        plt.close()
    """

def table_backtest(config, algos, labels=None, format="raw",
                   indicators=list(INDICATORS.keys())):
    """
    @:param config: config dictionary
    @:param algos: list of strings representing the name of algorithms
    or index of pgportfolio result
    @:param format: "raw", "html", "latex" or "csv". If it is "csv",
    the result will be save in a csv file. otherwise only print it out
    @:return: a string of html or latex code
    """
    results = []
    labels = list(labels)
    for i, algo in enumerate(algos):
        if algo.isdigit():
            portfolio_changes = _load_from_summary(algo, config, "result")
            logging.info("load index " + algo + " from csv file")
        else:
            logging.info("start executing " + algo)
            portfolio_changes = execute_backtest(algo, config)
            logging.info("finish executing " + algo)

        indicator_result = {}
        #calculates for every algo the indicators
        for indicator in indicators:
            indicator_result[indicator] = INDICATORS[indicator](portfolio_changes)
        results.append(indicator_result)
        if len(labels)<=i:
            labels.append(NAMES[algo])

    dataframe = pd.DataFrame(results, index=labels)

    start, end = _extract_test(config)
    start = datetime.datetime.fromtimestamp(start - start%config["input"]["global_period"])
    end = datetime.datetime.fromtimestamp(end - end%config["input"]["global_period"])

    print("backtest start from "+ str(start) + " to " + str(end))
    if format == "html":
        print(dataframe.to_html())
    elif format == "latex":
        print(dataframe.to_latex())
    elif format == "raw":
        print(dataframe.to_string())
    elif format == "csv":
        dataframe.to_csv("./compare"+end.strftime("%Y-%m-%d")+".csv")
    else:
        raise ValueError("The format " + format + " is not supported")

#returns start and end of testing period
def _extract_test(config):
    global_start = parse_time(config["input"]["start_date"])
    global_end = parse_time(config["input"]["end_date"])
    span = global_end - global_start
    start = global_end - config["input"]["test_portion"] * span
    end = global_end
    return start, end

#returns the backtest test history (portfolio vector change) of backtesting as a string, the last change
def _load_from_summary(index, config, what):
    """ load the backtest result form train_package/train_summary
    @:param index: index of the training and backtest
    @:return: numpy array of the portfolio changes
    """
    dataframe = pd.DataFrame.from_csv("./train_package/train_summary.csv")
    history_string = dataframe.loc[int(index)]["backtest_test_history"]
    #start date and end date and test period must be the same!
    if not check_input_same(config, json.loads(dataframe.loc[int(index)]["config"])):
        raise ValueError("the date of this index is not the same as the default config")
    if what == "result":
        return np.fromstring(history_string, sep=",")[:-1]
    if what == "weight":
        json_dir = "./train_package/" + str(int(index)) + "/weight_history" + str(int(index)) + ".json"
        with open(json_dir) as json_file:
            data = json.load(json_file)
        return data["portfolioweights"]

def get_panel(start, end, period=300, coinlist=None):
    if coinlist is None:
        return None
    DATABASE_DIR="database/Data.db"
    connection = sqlite3.connect(DATABASE_DIR)
    feature = "volume"
    try:
        for row_number, coin in enumerate(coinlist):
            if feature=="close":
                sql = ("SELECT date+300 AS date_norm, close FROM History WHERE"
                               " date_norm>={start} and date_norm<={end}" 
                               " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                serial_data = pd.read_sql_query(sql, con=connection)
            if feature == "volume":
                sql = ("SELECT date_norm, SUM(volume)" +
                   " FROM (SELECT date+{period}-(date%{period}) "
                   "AS date_norm, volume, coin FROM History)"
                   " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                   " GROUP BY date_norm".format(
                       period=period, start=start, end=end, coin=coin))
                serial_data = pd.read_sql_query(sql, con=connection)
    finally:
        connection.commit()
        connection.close()
    return serial_data