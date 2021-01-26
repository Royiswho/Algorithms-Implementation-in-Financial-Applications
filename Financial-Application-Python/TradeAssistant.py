"""
Implemented algorithms
divide and conquer, sorting for computing max drawdown of a stock and multithreading for speeding up computing
Greedy algorithm for calculating the maximal profit that is made by trading (long one share or not) a stock without commission
Dynamic programming for calculating the maximal profit that is made by trading (long one share or not) a stock with commission
author: Yi Rong
updated on 1/25/21
"""


import pandas_datareader as web
import datetime as dt
from pandas_datareader._utils import RemoteDataError
import concurrent.futures
import matplotlib.pyplot as plt


class TradeAssistant:

    ################## TradeAssistant initialization ##################
    def __init__(self, L_symbol, start, end):
        self.L_symbol = L_symbol
        self.start = start
        self.end = end
        L_data = self.get_raw_data()
        self.dict_raw_data = L_data[0]
        self.L_ts = L_data[1]
        self.fee = 1

    def get_raw_data(self):
        """
        Download raw data from yahoo finance and return all results in a dictionary
        """
        dict_raw_data = {}
        L_ts = []
        for symbol in self.L_symbol:
            try:
                df_symbol = web.get_data_yahoo(symbol, start=self.start, end=self.end)
                dict_raw_data[symbol] = df_symbol['Adj Close'].round(5)
            except RemoteDataError:
                print("No data fetched for symbol" + symbol)

            L_ts.append(dict_raw_data[symbol])
        return [dict_raw_data, L_ts]

    ################## MDD ##################
    # Step 1
    def get_cummax(self, ts):
        """
        Intent: get a cumulative max series and index of max.
                returnL is a list denotes the cumulative max of self.ts and its index, which means
                there are two values in each element. first value is the cumulative max, the second value is its index.
                e.g. self.ts = [10, 9, 8, 9, 12] ==>
                     returnL = [[10, 0], [10, 0], [10, 0], [10, 0], [12, 4]]

        Precondition: The global variable self.ts is the time series input of this
                      function.

        Postcondition1: (max) returnL[i][0] >= self.ts[j], for j in [0, i];
                        (index) returnL[i][1] <= i;
                        i can be any integer among [0, len(self.ts) - 1]
        """
        i, i_max = 0, 0
        returnL = []
        cur_max = ts[i]
        while i < len(ts):
            if cur_max < ts[i]:
                cur_max = ts[i]
                i_max = i
            returnL.append([cur_max, i_max])
            i += 1
        return returnL

    # Step 2: Divide and Conquer
    def get_mdd_between(self, ts, L_cummax, a_begin, an_end):
        """
        Intent: find the minimal value among a drawdown series and the start and end index of the max drawdown using divide and conquer

        Shorthand: L = len(ts)
        Precondition 1: ts is a list or a time series of floats
        Precondition 2: L_cummax is a list denotes the cumulative max of ts and its index, which means
        there are two values in each element. first value is the cumulative max, the second value is its index.
        e.g. ts = [10, 9, 8, 9, 12] ==> L_cummax = [[10, 0], [10, 0], [10, 0], [10, 0], [12, 4]]
        e.g. ts & L_cummax  ==> returnL = [0, 2, -20%]
        Pre3: 0 <= a_begin < L

        Postcondition 1 (Subsequence): 0 <= returnL[0] <= returnL[1] < L
        Post2 (L_cummax): L_cummax[a_begin][1] <= a_begin
        Post3 (Minimal): returnL[2] is minimal
        Post4 (max drawdown Constraint): 0 >= returnL[2] >= -1
        """

        returnL = [None] * 3

        """
        ===Sa (Solvable Immediately?):
        an_end_index = a_begin_index & Postcondition 1-3 & this returned
        –XOR–
        mid = int((a_begin + an_end) / 2)        
        """
        if a_begin == an_end:  # immediately satisfy all postconditions
            returnL[0] = L_cummax[a_begin][1]
            returnL[1] = a_begin
            returnL[2] = ts[a_begin] / L_cummax[a_begin][0] - 1
            return returnL
        else:
            mid = int((a_begin + an_end) / 2)

        # ===Sb1: Postcondition holds on a_list[:mid+1]?
        left_res = self.get_mdd_between(ts, L_cummax, a_begin, mid)

        # ===Sb2: Postcondition holds on a_list[mid+1:]
        right_res = self.get_mdd_between(ts, L_cummax, mid + 1, an_end)

        # ===Sc (Conquered) = Postconditions

        # Satisfy Pos3
        if left_res[2] < right_res[2]:
            return left_res

        else:
            return right_res

    # Greedy algorithm
    def get_max_profit(self, ts):
        """
        intent: get the maximal profit that can be made by making transactions based on a time series stock price data,
        there is no limit on the number transactions, but you can only hold one or zero share of stock at any time
        Precondition1: ts is a list or a time series of floats
        Precondition2 (positive price): ts[i] > 0 for all i

        Postcondition1: return_res >= 0
        Postcondition2: return_res is maximal
        """

        #Sa: ts is an empty list, 0 is returned
        if len(ts) == 0:
            return 0

        #Sb (Parts): return_res is the solution for ts[:i] for i in [1, len(ts)]
        #Sc (Greed used): return_res starts from 0 and will add positive difference: ts[i] - ts[i - 1], i in [1, len(ts))
        #Example: ts = [1, 3, 2, 4, 3],
        #     lag(ts)=    [1, 3, 2, 4]
        #differences =    [2,-1, 2,-1], so return_res = 0 + 2 + 2
        return_res = 0
        for i in range(1, len(ts)):
            if ts[i] - ts[i - 1] > 0:
                return_res += (ts[i] - ts[i - 1])

        # Sd: i is maximal, return_res is divided by the first-day price to make it a profit ratio
        return_res = return_res / ts[0]
        return return_res

    # DP
    def get_max_profit_with_transactionfee(self, ts):
        """
        intent: get the maximal profit that can be made by making transactions based on a time series stock price data,
        you can only hold one or zero share of stock at any time and you need to pay a transaction fee for each transaction,
        buy and sell refer to one transaction.
        Precondition1: ts is a list or a time series of floats
        Precondition2 (positive price): ts[i] > 0 for all i
        Precondition3 (positive fee): self.fee is a float number and fee > 0

        Postcondition1: return_res >= 0
        Postcondition2: return_res is maximal
        """

        #Sa: ts is an empty list, 0 is returned
        if len(ts) == 0:
            return 0

        #Sb: cash_ is the profit that we hold cash instead of holding stock, hold_ is the profit that we hold stock
        # cash_ and hold_ are initialized for the first day
        cash_ = 0
        hold_ = -ts[0]

        # aProblem = The maximal profit of holding cash or a stock that is transited from the last cash_ and hold_.
        def _getDPSol(cash_, hold_):
            """
            Precondition1: the solution for each day is between cash_ and hold_, which are knownS

            Postcondition1: knownS contains a solution for aProblem
            """
            #Sa (transition): on day i, the max profit of holding cash or a stock is transitted from day i -1.
            for i in range(1, len(ts)):
                # Sb1: on day i, our max profit of holding cash can be determined between keeping cash or selling a stock
                #       keep cash, sell stock and pay transaction fee
                cash_ = max(cash_, hold_ + ts[i] - self.fee)
                # Sb2: on day i, our max profit of holding a stock can be determined between keeping the stock or buying a stock
                #      keep stock, buy a stock
                hold_ = max(hold_, cash_ - ts[i])

            # Sc (completed): on the last day, cash_ is the solution
            return cash_

        # Sc: the max profit is divided by the first-day price to make it a profit ratio
        return_res = _getDPSol(cash_, hold_) / ts[0]
        return return_res


    def get_analysis(self, ts):
        """
        intent: combine get_cummax, get_mdd_between, get_max_profit in one function so that this function can be used in multithreading

        Precondition 1: ts is a list or a time series of floats

        Postcondition 1 (Subsequence): 0 <= returnL[0] <= returnL[1] < L
        Post2 (Minimal): returnL[2] is minimal
        Post3 (max drawdown Constraint): 0 >= returnL[2] >= -1
        """
        L_cummax = self.get_cummax(ts)
        return_res = self.get_mdd_between(ts, L_cummax, 0, (len(L_cummax) - 1))
        return_res.append(self.get_max_profit(ts))
        return_res.append(self.get_max_profit_with_transactionfee(ts))

        return return_res

    def do_multithreading(self):
        """
        intent: get max drawdown results for multiple time series using multithreading

        Precondition 1: self.L_ts is a list, its element is a time series of floats
        Pre2: self.get_analysis is a function that can get analysis results like
                max drawdown, max profit, max profit with transaction feefor each element in self.L_ts

        Postcondition 1: len(return_L_mdd) = len(self.L_ts)
        Post2: the order of return_L_mdd should be the same as self.L_ts, which is also the order of self.L_symbol
        Post3: each list element of return_L_mdd has its corresponding symbol as the fourth element.
                return_L_mdd[i][3] = self.L_symbol[i] for i in [0, len(self.L_ts) - 1]
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # ---S1 Thread self.get_analysis(self.L_ts[0]) started, which attains results_[0] = self.get_analysis(self.L_ts[0])
            # ---S2 Thread self.get_analysis(self.L_ts[1]) started, which attains results_[1] = self.get_analysis(self.L_ts[1])
            # ...
            # ---Sn Thread self.get_analysis(self.L_ts[n - 1]) started, which attains results_[n - 1] = self.get_analysis(self.L_ts[n - 1])
            # ---Total number of len(self.L_ts) threads started
            results_ = executor.map(self.get_analysis, self.L_ts)
            """
            executor.map will start threads recursively for each element in self.L_ts and put it in function self.get_analysis.
            One advantage of this function is the return results_ will have the same order as self.L_ts, rather than the order based on computaional efficiency.
            """
            # ---Sn+1: Threads in S1, S2, ..., Sn completed
            # ---Sn+2: results_[i] = self.get_analysis(self.L_ts[i]) for i in [0, len(self.L_ts) - 1]

            # ---Sn+3: for each list element in results_, add its corresponding symbol as the fourth element in the list element
            return_L_mdd = []
            for i, result in enumerate(results_):
                result.append(self.L_symbol[i])
                return_L_mdd.append(result)

        return return_L_mdd

    def merge_sort(self, a_list, col):
        """
        intent: a_list is a list in which each element is a list, this function will sort the a_list based on a certain element is the list element, which refers to a_list[i][col] for i in [0, len(a_list) - 1]

        precondition1: len(a_list) < 0
        pre2: len(a_list[i]) is the same for i in [0, len(a_list) - 1] and len(a_list[i]) >= (col + 1)
        pre3: -1 <= a_list[i][col] <= 0 for i in [0, len(a_list) - 1]

        postcondition1: return_list will be a sorted version of a_list, which means if len(return_list) > 2
                        return_list[i][col] <= return_list[i + 1][col] <= return_list[i + 2][col] <= ...
        post2: return_list is the same multiset of a_list
        """

        # ---Sa (Solvable Immediately?):
        # len(a_list) == 1 & Postconditions & This returned
        # –XOR–
        # mid = int(len(a_list) / 2)

        # if not immediately satisfy all postconditions?
        if len(a_list) > 1:

        # then
            mid = int(len(a_list) / 2)

            # ---Sb1: Postconditions hold on a_list[:mid]
            left = a_list[:mid].copy()
            self.merge_sort(left, col)

            # ---Sb2: Postconditions hold on a_list[mid:]
            right = a_list[mid:].copy()
            self.merge_sort(right, col)

            # ---Sc (Conquered) = Postconditions
            l = 0 # index in left
            r = 0 # index in right
            k = 0 # index in a_list and a_list[:mid] = left, a_list[mid:] = right

            # ---Sc1 (some sorted): a_list[:k] is sorted when l < len(left) and r < len(right)
            #                       and k = l + r
            while l < len(left) and r < len(right):
                if left[l][col] <= right[r][col]:
                    a_list[k] = left[l]
                    l += 1
                    k += 1
                else:
                    a_list[k] = right[r]
                    r += 1
                    k += 1

            # ---Sc2 (some sorted): a_list[:k] is sorted when r = len(right), which means all elements in right is sorted and placed in a_list
            #                       and k = l + r
            while l < len(left):
                a_list[k] = left[l]
                l += 1
                k += 1

            # ---Sc3 (some sorted): a_list[:k] is sorted when l = len(left), which means all elements in left is sorted and placed in a_list
            #                       and k = l + r
            while r < len(right):
                a_list[k] = right[r]
                r += 1
                k += 1

        # ---S4 (Complement): a_list is sorted
        return_list = a_list.copy()
        return return_list



    ################## main ##################
    def main(self):
        L_mdd = self.do_multithreading()
        a_list = self.merge_sort(L_mdd, col=2)
        print(" Sorted Max Drawdown: ")
        for i, res in enumerate(a_list):
            # pdb.set_trace()
            print(f"---------------------- No.{i + 1} -------------------------------")
            print(f"The max drawdown of {res[-1]} is " + "{:.2%}".format(res[2]))
            print(f"The max drawdown starts on  {str(self.dict_raw_data[res[-1]].index[res[0]])}")
            print(f"The max drawdown ends on {str(self.dict_raw_data[res[-1]].index[res[1]])}")
            print(f"The max profit can be " + "{:.2%}".format(res[3]))
            print(f"The max profit with transaction fee $" + str(self.fee) + " can be " + "{:.2%}".format(res[4]))
            # self.dict_raw_data[res[3]].plot(color='grey', figsize=(8,4))
            # plt.title(res[3])
            # plt.ylabel("Adj Close")
            #
            # plt.scatter(self.dict_raw_data[res[3]].index[res[0]],  self.dict_raw_data[res[3]][res[0]], color='red')
            # plt.scatter(self.dict_raw_data[res[3]].index[res[1]],  self.dict_raw_data[res[3]][res[1]], color='red')
            # plt.show()


if __name__ == "__main__":
    L_symbol = ["TSLA", "FB", "NFLX", "AMZN", "GOOG"]
    TA = TradeAssistant(L_symbol=L_symbol, start=dt.datetime(2018, 1, 1), end=dt.datetime(2020, 7, 1))
    TA.main()
