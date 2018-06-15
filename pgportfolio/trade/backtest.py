from __future__ import absolute_import, division, print_function
import numpy as np
from pgportfolio.trade import trader
from pgportfolio.marketdata.datamatrices import DataMatrices
import logging
from pgportfolio.tools.trade import calculate_pv_after_commission


class BackTest(trader.Trader):
    def __init__(self, config, net_dir=None, agent=None, agent_type="nn"):
        #initialize Trader class in trader
        trader.Trader.__init__(self, 0, config, 0, net_dir,
                               initial_BTC=1, agent=agent, agent_type=agent_type)
        if agent_type == "nn":
            data_matrices = self._rolling_trainer.data_matrices
        elif agent_type == "traditional":
            config["input"]["feature_number"] = 1
            data_matrices = DataMatrices.create_from_config(config)
        else:
            raise ValueError()
        #gets test set from datamatrices
        self.__test_set = data_matrices.get_test_set()
        self.__test_length = self.__test_set["X"].shape[0]
        self._total_steps = self.__test_length
        self.__test_pv = 1.0
        self.__test_pc_vector = []
        self.__test_updated_omega= []

    #used from tradertrainer during result collection, returns total capital
    @property
    def test_pv(self):
        return self.__test_pv

    #used from tradetrainer during result collection, returns vector of portfolio changes
    @property
    def test_pc_vector(self):
        return np.array(self.__test_pc_vector, dtype=np.float32)

    @property
    def test_updated_omega(self):
        return self.__test_updated_omega

    #startet by trader after backtesting, sets total capital
    def finish_trading(self):
        self.__test_pv = self._total_capital

        """
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(self._rolling_trainer.data_matrices.sample_count)),
               self._rolling_trainer.data_matrices.sample_count)
        fig.tight_layout()
        plt.show()
        """

    def _log_trading_info(self, time, omega):
        pass

    def _initialize_data_base(self):
        pass

    def _write_into_database(self):
        pass

    #returns the latest window of X of the test set at the current time point during backtesting
    def __get_matrix_X(self):
        return self.__test_set["X"][self._steps]

    #used in trade_by_strategy, returns the latest relative Prices of the window at the current time point
    # during backtesting
    def __get_matrix_y(self):
        return self.__test_set["y"][self._steps, 0, :]

    def rolling_train(self, online_sample=None):
        self._rolling_trainer.rolling_train()

    #returns the latest window of X of the test set until the current time point during backtesting
    def generate_history_matrix(self):
        inputs = self.__get_matrix_X()
        if self._agent_type == "traditional":
            inputs = np.concatenate([np.ones([1, 1, inputs.shape[2]]), inputs], axis=1)
            inputs = inputs[:, :, 1:] / inputs[:, :, :-1]
        return inputs

    #started by trader, after new weights for the current time step got calculated
    def trade_by_strategy(self, omega):
        logging.info("the step is {}".format(self._steps))
        logging.debug("the raw omega is {}".format(omega))
        #vector of ones and the relative future price
        future_price = np.concatenate((np.ones(1), self.__get_matrix_y()))
        #returns a numerically calculated commission
        pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        #portfolio change
        portfolio_change = pv_after_commission * np.dot(omega, future_price)
        self._total_capital *= portfolio_change
        #set the new last omega to be equal to the new portfolio after commission and price changes
        self._last_omega = pv_after_commission * omega * \
                           future_price /\
                           portfolio_change
        logging.debug("the new omega is {}".format(self._last_omega))
        logging.debug("the portfolio change this period is : {}".format(portfolio_change))
        self.__test_pc_vector.append(portfolio_change)
        self.__test_updated_omega.append(self._last_omega)

