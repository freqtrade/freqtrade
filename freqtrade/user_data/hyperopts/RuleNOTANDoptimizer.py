# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension,Integer , Real  # noqa
from freqtrade.optimize.space import SKDecimal
from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib

##PYCHARM
import sys
sys.path.append(r"/freqtrade/user_data/strategies")


# ##HYPEROPT
# import sys,os
# file_dir = os.path.dirname(__file__)
# sys.path.append(file_dir)


from  z_buyer_mid_volatility import mid_volatility_buyer
from  z_seller_mid_volatility import mid_volatility_seller
from  z_COMMON_FUNCTIONS import MID_VOLATILITY




class RuleOptimizer15min(IHyperOpt):
    """
    This is a sample hyperopt to inspire you.
    Feel free to customize it.

    More information in the documentation: https://www.freqtrade.io/en/latest/hyperopt/

    You should:
    - Rename the class name to some unique name.
    - Add any methods you want to build your hyperopt.
    - Add any lib you need to build your hyperopt.

    You must keep:
    - The prototypes for the methods: populate_indicators, indicator_space, buy_strategy_generator.

    The methods roi_space, generate_roi_table and stoploss_space are not required
    and are provided by default.
    However, you may override them if you need the
    'roi' and the 'stoploss' spaces that differ from the defaults offered by Freqtrade.

    This sample illustrates how to override these methods.
    """


    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by hyperopt
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use
            """
            conditions = []



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


            ##MAIN SELECTORS

#--------------------

            ##VOLATILITY

            conditions.append(dataframe['vol_mid']  > 0  )

           # conditions.append((dataframe['vol_low']  > 0) |(dataframe['vol_mid']  > 0) )

            # conditions.append((dataframe['vol_high']  > 0) |(dataframe['vol_mid']  > 0) )


#--------------------


            ##PICKS TREND COMBO

            conditions.append(

                (dataframe['downtrend'] >= params['main_1_trend_strength'])
                |#OR &
                (dataframe['downtrendsmall'] >= params['main_2_trend_strength'])

            )

            ##UPTREND
            #conditions.append(dataframe['uptrend'] >= params['main_1_trend_strength'])
            ##DOWNTREND
            #conditions.append(dataframe['downtrend'] >= params['main_1_trend_strength'])
            ##NOTREND
            #conditions.append((dataframe['uptrend'] <1)&(dataframe['downtrend'] <1))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            ##ABOVE / BELOW THRESHOLDS

            #RSI ABOVE
            if 'include_sell_ab_9_rsi_above_value' in params and params['include_sell_ab_9_rsi_above_value']:
                conditions.append(dataframe['rsi'] > params['sell_ab_9_rsi_above_value'])
            #RSI RECENT PIT 5
            if 'include_sell_ab_10_rsi_recent_pit_2_value' in params and params['include_sell_ab_10_rsi_recent_pit_2_value']:
                conditions.append(dataframe['rsi'].rolling(2).min() < params['sell_ab_10_rsi_recent_pit_2_value'])
            #RSI RECENT PIT 12
            if 'include_sell_ab_11_rsi_recent_pit_4_value' in params and params['include_sell_ab_11_rsi_recent_pit_4_value']:
                conditions.append(dataframe['rsi'].rolling(4).min() < params['sell_ab_11_rsi_recent_pit_4_value'])
            #RSI5 BELOW
            if 'include_sell_ab_12_rsi5_above_value' in params and params['include_sell_ab_12_rsi5_above_value']:
                conditions.append(dataframe['rsi5'] > params['sell_ab_12_rsi5_above_value'])
            #RSI50 BELOW
            if 'include_sell_ab_13_rsi50_above_value' in params and params['include_sell_ab_13_rsi50_above_value']:
                conditions.append(dataframe['rsi50'] > params['sell_ab_13_rsi50_above_value'])

#-----------------------

            #ROC BELOW
            if 'include_sell_ab_14_roc_above_value' in params and params['include_sell_ab_14_roc_above_value']:
                conditions.append(dataframe['roc'] > (params['sell_ab_14_roc_above_value']/2))
            #ROC50 BELOW
            if 'include_sell_ab_15_roc50_above_value' in params and params['include_sell_ab_15_roc50_above_value']:
                conditions.append(dataframe['roc50'] > (params['sell_ab_15_roc50_above_value']))
            #ROC2 BELOW
            if 'include_sell_ab_16_roc2_above_value' in params and params['include_sell_ab_16_roc2_above_value']:
                conditions.append(dataframe['roc2'] > (params['sell_ab_16_roc2_above_value']/2))

#-----------------------

            #PPO5 BELOW
            if 'include_sell_ab_17_ppo5_above_value' in params and params['include_sell_ab_17_ppo5_above_value']:
                conditions.append(dataframe['ppo5'] > (params['sell_ab_17_ppo5_above_value']/2))
            #PPO10 BELOW
            if 'include_sell_ab_18_ppo10_above_value' in params and params['include_sell_ab_18_ppo10_above_value']:
                conditions.append(dataframe['ppo10'] > (params['sell_ab_18_ppo10_above_value']/2))
            #PPO25 BELOW
            if 'include_sell_ab_19_ppo25_above_value' in params and params['include_sell_ab_19_ppo25_above_value']:
                conditions.append(dataframe['ppo25'] > (params['sell_ab_19_ppo25_above_value']/2))

            #PPO50 BELOW
            if 'include_sell_ab_20_ppo50_above_value' in params and params['include_sell_ab_20_ppo50_above_value']:
                conditions.append(dataframe['ppo50'] > (params['sell_ab_20_ppo50_above_value']/2))
            #PPO100 BELOW
            if 'include_sell_ab_21_ppo100_above_value' in params and params['include_sell_ab_21_ppo100_above_value']:
                conditions.append(dataframe['ppo100'] > (params['sell_ab_21_ppo100_above_value']))
            #PPO200 BELOW
            if 'include_sell_ab_22_ppo200_above_value' in params and params['include_sell_ab_22_ppo200_above_value']:
                conditions.append(dataframe['ppo200'] > (params['sell_ab_22_ppo200_above_value']))
            #PPO500 BELOW
            if 'include_sell_ab_23_ppo500_above_value' in params and params['include_sell_ab_23_ppo500_above_value']:
                conditions.append(dataframe['ppo500'] > (params['sell_ab_23_ppo500_above_value']*2))

            ##USE AT A LATER STEP

            #convsmall BELOW
            if 'include_sell_ab_24_convsmall_above_value' in params and params['include_sell_ab_24_convsmall_above_value']:
                conditions.append(dataframe['convsmall'] > (params['sell_ab_24_convsmall_above_value']/2))
            #convmedium BELOW
            if 'include_sell_ab_25_convmedium_above_value' in params and params['include_sell_ab_25_convmedium_above_value']:
                conditions.append(dataframe['convmedium'] >(params['sell_ab_25_convmedium_above_value']))
            #convlarge BELOW
            if 'include_sell_ab_26_convlarge_above_value' in params and params['include_sell_ab_26_convlarge_above_value']:
                conditions.append(dataframe['convlarge'] > (params['sell_ab_26_convlarge_above_value']))
            #convultra BELOW
            if 'include_sell_ab_27_convultra_above_value' in params and params['include_sell_ab_27_convultra_above_value']:
                conditions.append(dataframe['convultra'] > (params['sell_ab_27_convultra_above_value']/2))
            #convdist BELOW
            if 'include_sell_ab_28_convdist_above_value' in params and params['include_sell_ab_28_convdist_above_value']:
                conditions.append(dataframe['convdist'] > (params['sell_ab_28_convdist_above_value']))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            ##SMA'S GOING DOWN

            if 'sell_down_0a_sma3' in params and params['sell_down_0a_sma3']:
                conditions.append((dataframe['sma3'].shift(1) >dataframe['sma3']))
            if 'sell_down_0b_sma5' in params and params['sell_down_0b_sma5']:
                conditions.append((dataframe['sma5'].shift(1) >dataframe['sma5']))
            if 'sell_down_1_sma10' in params and params['sell_down_1_sma10']:
                conditions.append((dataframe['sma10'].shift(1) >dataframe['sma10']))
            if 'sell_down_2_sma25' in params and params['sell_down_2_sma25']:
                conditions.append((dataframe['sma25'].shift(1) >dataframe['sma25']))
            if 'sell_down_3_sma50' in params and params['sell_down_3_sma50']:
                conditions.append((dataframe['sma50'].shift(2) >dataframe['sma50']))
            if 'sell_down_4_sma100' in params and params['sell_down_4_sma100']:
                conditions.append((dataframe['sma100'].shift(3) >dataframe['sma100']))
            if 'sell_down_5_sma200' in params and params['sell_down_5_sma200']:
                conditions.append((dataframe['sma200'].shift(4) >dataframe['sma200']))

            if 'sell_down_6_sma400' in params and params['sell_down_6_sma400']:
                conditions.append((dataframe['sma400'].shift(4) >dataframe['sma400']))
            if 'sell_down_7_sma10k' in params and params['sell_down_7_sma10k']:
                conditions.append((dataframe['sma10k'].shift(5) >dataframe['sma10k']))
            # if 'sell_down_8_sma20k' in params and params['sell_down_8_sma20k']:
            #     conditions.append((dataframe['sma20k'].shift(5) >dataframe['sma20k']))
            # if 'sell_down_9_sma30k' in params and params['sell_down_9_sma30k']:
            #     conditions.append((dataframe['sma30k'].shift(5) >dataframe['sma30k']))

            if 'sell_down_10_convsmall' in params and params['sell_down_10_convsmall']:
                conditions.append((dataframe['convsmall'].shift(2) >dataframe['convsmall']))
            if 'sell_down_11_convmedium' in params and params['sell_down_11_convmedium']:
                conditions.append((dataframe['convmedium'].shift(3) >dataframe['convmedium']))
            if 'sell_down_12_convlarge' in params and params['sell_down_12_convlarge']:
                conditions.append((dataframe['convlarge'].shift(4) >dataframe['convlarge']))
            if 'sell_down_13_convultra' in params and params['sell_down_13_convultra']:
                conditions.append((dataframe['convultra'].shift(4) >dataframe['convultra']))
            if 'sell_down_14_convdist' in params and params['sell_down_14_convdist']:
                conditions.append((dataframe['convdist'].shift(4) >dataframe['convdist']))

            if 'sell_down_15_vol50' in params and params['sell_down_15_vol50']:
                conditions.append((dataframe['vol50'].shift(2) >dataframe['vol50']))
            if 'sell_down_16_vol100' in params and params['sell_down_16_vol100']:
                conditions.append((dataframe['vol100'].shift(3) >dataframe['vol100']))
            if 'sell_down_17_vol175' in params and params['sell_down_17_vol175']:
                conditions.append((dataframe['vol175'].shift(4) >dataframe['vol175']))
            if 'sell_down_18_vol250' in params and params['sell_down_18_vol250']:
                conditions.append((dataframe['vol250'].shift(4) >dataframe['vol250']))
            if 'sell_down_19_vol500' in params and params['sell_down_19_vol500']:
                conditions.append((dataframe['vol500'].shift(4) >dataframe['vol500']))

            if 'sell_down_20_vol1000' in params and params['sell_down_20_vol1000']:
                conditions.append((dataframe['vol1000'].shift(4) >dataframe['vol1000']))
            if 'sell_down_21_vol100mean' in params and params['sell_down_21_vol100mean']:
                conditions.append((dataframe['vol100mean'].shift(4) >dataframe['vol100mean']))
            if 'sell_down_22_vol250mean' in params and params['sell_down_22_vol250mean']:
                conditions.append((dataframe['vol250mean'].shift(4) >dataframe['vol250mean']))

            if 'up_20_conv3' in params and params['up_20_conv3']:
                conditions.append(((dataframe['conv3'].shift(25) < dataframe['conv3'])&(dataframe['conv3'].shift(50) < dataframe['conv3'])))
            if 'up_21_vol5' in params and params['up_21_vol5']:
                conditions.append(((dataframe['vol5'].shift(25) < dataframe['vol5'])&(dataframe['vol5'].shift(50) < dataframe['vol5'])))
            if 'up_22_vol5ultra' in params and params['up_22_vol5ultra']:
                conditions.append(((dataframe['vol5ultra'].shift(25) < dataframe['vol5ultra'])&(dataframe['vol5ultra'].shift(50) < dataframe['vol5ultra'])))
            if 'up_23_vol1ultra' in params and params['up_23_vol1ultra']:
                conditions.append(((dataframe['vol1ultra'].shift(25) < dataframe['vol1ultra'])& (dataframe['vol1ultra'].shift(50) < dataframe['vol1ultra'])))
            if 'up_24_vol1' in params and params['up_24_vol1']:
                conditions.append(((dataframe['vol1'].shift(30) < dataframe['vol1'])&(dataframe['vol1'].shift(10) < dataframe['vol1'])))
            if 'up_25_vol5inc24' in params and params['up_25_vol5inc24']:
                conditions.append((dataframe['vol5inc24'].shift(50) < dataframe['vol5inc24']))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            ##ABOVE / BELOW SMAS    1 above/ 0 None  / -1 below

            #SMA10
            conditions.append((dataframe['close'] > dataframe['sma10'])|(0.5 > params['ab_1_sma10']))
            conditions.append((dataframe['close'] < dataframe['sma10'])|(-0.5 < params['ab_1_sma10']))
            #SMA25
            conditions.append((dataframe['close'] > dataframe['sma25'])|(0.5 > params['ab_2_sma25']))
            conditions.append((dataframe['close'] < dataframe['sma25'])|(-0.5 < params['ab_2_sma25']))
            #SMA50
            conditions.append((dataframe['close'] > dataframe['sma50'])|(0.5 > params['ab_3_sma50']))
            conditions.append((dataframe['close'] < dataframe['sma50'])|(-0.5 < params['ab_3_sma50']))


            #SMA100
            conditions.append((dataframe['close'] > dataframe['sma100'])|(0.5 > params['ab_4_sma100']))
            conditions.append((dataframe['close'] < dataframe['sma100'])|(-0.5 < params['ab_4_sma100']))
            #SMA100
            conditions.append((dataframe['close'] > dataframe['sma200'])|(0.5 > params['ab_5_sma200']))
            conditions.append((dataframe['close'] < dataframe['sma200'])|(-0.5 < params['ab_5_sma200']))
            #SMA400
            conditions.append((dataframe['close'] > dataframe['sma400'])|(0.5 > params['ab_6_sma400']))
            conditions.append((dataframe['close'] < dataframe['sma400'])|(-0.5 < params['ab_6_sma400']))
            #SMA10k
            conditions.append((dataframe['close'] > dataframe['sma10k'])|(0.5 > params['ab_7_sma10k']))
            conditions.append((dataframe['close'] < dataframe['sma10k'])|(-0.5 < params['ab_7_sma10k']))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            ##DOWNSWINGS / UPSWINGS PPO'S

            #ppo5  UP OR DOWN (1 UP, 0 NOTHING, -1 DOWN)
            conditions.append((dataframe['ppo5'].shift(2) <dataframe['ppo5'])|(0.5 > params['sell_swings_1_ppo5_up_or_down_bool']))
            conditions.append((dataframe['ppo5'].shift(2) >dataframe['ppo5'])|(-0.5 < params['sell_swings_1_ppo5_up_or_down_bool']))
            #ppo10
            conditions.append((dataframe['ppo10'].shift(3) <dataframe['ppo10'])|(0.5 > params['sell_swings_2_ppo10_up_or_down_bool']))
            conditions.append((dataframe['ppo10'].shift(3) >dataframe['ppo10'])|(-0.5 < params['sell_swings_2_ppo10_up_or_down_bool']))
            #ppo25
            #conditions.append((dataframe['ppo25'].shift(3) <dataframe['ppo25'])|(0.5 > params['sell_swings_3_ppo25_up_or_down_bool']))
            conditions.append((dataframe['ppo25'].shift(3) >dataframe['ppo25'])|(-0.5 < params['sell_swings_3_ppo25_up_or_down_bool']))

            #ppo50
            #conditions.append((dataframe['ppo50'].shift(3 <dataframe['ppo50'])|(0.5 > params['sell_swings_4_ppo50_up_or_down_bool']))
            conditions.append((dataframe['ppo50'].shift(3) >dataframe['ppo50'])|(-0.5 < params['sell_swings_4_ppo50_up_or_down_bool']))
            #ppo100
            #conditions.append((dataframe['ppo100'].shift(4) <dataframe['ppo100'])|(0.5 > params['sell_swings_5_ppo100_up_or_down_bool']))
            conditions.append((dataframe['ppo100'].shift(4) >dataframe['ppo100'])|(-0.5 < params['sell_swings_5_ppo100_up_or_down_bool']))
            #ppo200
            #conditions.append((dataframe['ppo200'].shift(4) <dataframe['ppo200'])|(0.5 > params['sell_swings_6_ppo200_up_or_down_bool']))
            conditions.append((dataframe['ppo200'].shift(4) >dataframe['ppo200'])|(-0.5 < params['sell_swings_6_ppo200_up_or_down_bool']))

            #ppo500
            #conditions.append((dataframe['ppo500'].shift(5) <dataframe['ppo500'])|(0.5 > params['sell_swings_7_ppo500_up_or_down_bool']))
            conditions.append((dataframe['ppo500'].shift(5) >dataframe['ppo500'])|(-0.5 < params['sell_swings_7_ppo500_up_or_down_bool']))

            #roc50
            #conditions.append((dataframe['roc50'].shift(3) <dataframe['roc50'])|(0.5 > params['sell_swings_8_roc50_up_or_down_bool']))
            conditions.append((dataframe['roc50'].shift(3) >dataframe['roc50'])|(-0.5 < params['sell_swings_8_roc50_up_or_down_bool']))
            #roc10
            #conditions.append((dataframe['roc10'].shift(2) <dataframe['roc10'])|(0.5 > params['sell_swings_9_roc10_up_or_down_bool']))
            conditions.append((dataframe['roc10'].shift(2) >dataframe['roc10'])|(-0.5 < params['sell_swings_9_roc10_up_or_down_bool']))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


            ##DISTANCES/ROC

                         ##FOR MORE TOP SELLERS
            #dist50 MORE THAN
            if 'include_sell_dist_1_dist50_more_value' in params and params['include_sell_dist_1_dist50_more_value']:
                conditions.append(dataframe['dist50'] > (params['sell_dist_1_dist50_more_value']))
            #dist200 MORE THAN
            if 'include_sell_dist_2_dist200_more_value' in params and params['include_sell_dist_2_dist200_more_value']:
                conditions.append(dataframe['dist200'] > (params['sell_dist_2_dist200_more_value']))

            #dist400 MORE THAN
            if 'include_sell_dist_3_dist400_more_value' in params and params['include_sell_dist_3_dist400_more_value']:
                conditions.append(dataframe['dist400'] > (params['sell_dist_3_dist400_more_value']))
            #dist10k MORE THAN
            if 'include_sell_dist_4_dist10k_more_value' in params and params['include_sell_dist_4_dist10k_more_value']:
                conditions.append(dataframe['dist10k'] > (params['sell_dist_4_dist10k_more_value']))

                        ##FOR MORE TOP SELLERS
            #more =further from top bol up
            #dist_upbol50 MORE THAN
            if 'include_sell_dist_5_dist_upbol50_more_value' in params and params['include_sell_dist_5_dist_upbol50_more_value']:
                conditions.append(dataframe['dist_upbol50'] > (params['sell_dist_5_dist_upbol50_more_value']/2))
            #dist_upbol100 MORE THAN
            if 'include_sell_dist_6_dist_upbol100_more_value' in params and params['include_sell_dist_6_dist_upbol100_more_value']:
                conditions.append(dataframe['dist_upbol100'] > (params['sell_dist_6_dist_upbol100_more_value']/2))


                        ##for bot bol prevent seller
            # #less =closer to bot bol
            #dist_upbol50 LESS THAN.
            #if 'include_sell_dist_7_dist_lowbol50_more_value' in params and params['include_sell_dist_7_dist_lowbol50_more_value']:
            #   conditions.append(dataframe['dist_lowbol50'] > (params['sell_dist_7_dist_lowbol50_more_value']/2))
            #dist_upbol100 LESS THAN
           # if 'include_sell_dist_8_dist_lowbol100_more_value' in params and params['include_sell_dist_8_dist_lowbol100_more_value']:
            #   conditions.append(dataframe['dist_lowbol100'] > (params['sell_dist_8_dist_lowbol100_more_value']/2))



            ##others
             #roc50sma LESS THAN
            if 'include_sell_dist_7_roc50sma_less_value' in params and params['include_sell_dist_7_roc50sma_less_value']:
                conditions.append(dataframe['roc50sma'] < (params['sell_dist_7_roc50sma_less_value'])*2)
            #roc200sma LESS THAN
            if 'include_sell_dist_8_roc200sma_less_value' in params and params['include_sell_dist_8_roc200sma_less_value']:
                conditions.append(dataframe['roc200sma'] < (params['sell_dist_8_roc200sma_less_value'])*2)

            ##ENABLE TO BUY AWAY FROM HIGH
            # #HIGH500 TO CLOSE  MORE THAN
            #if 'include_sell_dist_9_high100_more_value' in params and params['include_sell_dist_9_high100_more_value']:
            #   conditions.append((dataframe['high100']-dataframe['close']) > ((dataframe['high100']/100* (params['sell_dist_9_high100_more_value']))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






            # Check that volume is not 0
            conditions.append(dataframe['volume'] > 0)




            if conditions:


                # ##ENABLE PRODUCTION BUYS
                # dataframe.loc[
                #     (add_production_buys_mid(dataframe)),
                #     'buy'] = 1
                #


                dataframe.loc[
                    (~(reduce(lambda x, y: x & y, conditions)))&OPTIMIZED_RULE(dataframe,params),
                    'buy'] = 1

            return dataframe

        return populate_buy_trend

    @staticmethod
    def indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching strategy parameters
        """
        return [


#-------------------------------------------------------------------------------------------------------

            ## CUSTOM RULE TRESHOLDS

            # SKDecimal(5.0, 7.0,decimals=1, name='sell_trigger_0_roc_ab_value'),# 5       range 5-7  or 4-7
            # SKDecimal(3.2, 4.5,decimals=1, name='sell_trigger_0_roc2_ab_value'),#3.8   range 3.2-4.5
            # Integer(77, 82, name='sell_trigger_0_rsi_ab_value'),#79      range 82-77
            # Integer(90, 95, name='sell_trigger_0_rsi5_ab_value'),#94    range  95-90
            # Integer(63, 67, name='sell_trigger_0_rsi50_ab_value'),#66   range 67-63

#-------------------------------------------------------------------------------------------------------

            ##MAIN

            Categorical([1, 2, 3], name='sell_main_1_trend_strength'), #BIG TREND STR
            Categorical([1, 2, 3], name='sell_main_2_trend_strength'), #SMALL UPTREND STR


            #Categorical([-1, 0, 1], name='sell_main_2_small_uptrend_downtrend'), #SMALL UPTREND ON/OFF  1 is on -1 is down

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

             ##INCLUDE/EXCLUDE RULES

            Categorical([True, False], name='include_sell_ab_9_rsi_above_value'),
            Categorical([True, False], name='include_sell_ab_10_rsi_recent_pit_2_value'),
            Categorical([True, False], name='include_sell_ab_11_rsi_recent_pit_4_value'),
            Categorical([True, False], name='include_sell_ab_12_rsi5_above_value'),
            Categorical([True, False], name='include_sell_ab_13_rsi50_above_value'),

            Categorical([True, False], name='include_sell_ab_14_roc_above_value'),
            Categorical([True, False], name='include_sell_ab_15_roc50_above_value'),
            Categorical([True, False], name='include_sell_ab_16_roc2_above_value'),

            Categorical([True, False], name='include_sell_ab_17_ppo5_above_value'),
            Categorical([True, False], name='include_sell_ab_18_ppo10_above_value'),
            Categorical([True, False], name='include_sell_ab_19_ppo25_above_value'),

            Categorical([True, False], name='include_sell_ab_20_ppo50_above_value'),
            Categorical([True, False], name='include_sell_ab_21_ppo100_above_value'),
            Categorical([True, False], name='include_sell_ab_22_ppo200_above_value'),
            Categorical([True, False], name='include_sell_ab_23_ppo500_above_value'),

            ##USE AT A LATER STEP
            Categorical([True, False], name='include_sell_ab_24_convsmall_above_value'),
            Categorical([True, False], name='include_sell_ab_25_convmedium_above_value'),
            Categorical([True, False], name='include_sell_ab_26_convlarge_above_value'),
            Categorical([True, False], name='include_sell_ab_27_convultra_above_value'),
            Categorical([True, False], name='include_sell_ab_28_convdist_above_value'),

            Categorical([True, False], name='include_sell_dist_1_dist50_more_value'),
            Categorical([True, False], name='include_sell_dist_2_dist200_more_value'),
            Categorical([True, False], name='include_sell_dist_3_dist400_more_value'),
            Categorical([True, False], name='include_sell_dist_4_dist10k_more_value'),

            Categorical([True, False], name='include_sell_dist_5_dist_upbol50_more_value'),
            Categorical([True, False], name='include_sell_dist_6_dist_upbol100_more_value'),


            # FOR MORE DOWNTREND BUYS LIKELY
            # Categorical([True, False],  name='include_sell_dist_7_dist_lowbol50_more_value'),
            # Categorical([True, False],  name='include_sell_dist_8_dist_lowbol100_more_value'),

            #MORE LIKE TRIGGERS
            Categorical([True, False],  name='include_sell_dist_7_roc50sma_less_value'),
           Categorical([True, False],  name='include_sell_dist_8_roc200sma_less_value'),

            ##below high 100
            #Categorical([True, False],  name='include_sell_dist_9_high100_more_value'),

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

            ##ABOVE/BELOW VALUES

            Integer(35, 82, name='sell_ab_9_rsi_above_value'),
            Integer(18, 35, name='sell_ab_10_rsi_recent_pit_2_value'),
            Integer(18, 35, name='sell_ab_11_rsi_recent_pit_4_value'),
            Integer(70, 91, name='sell_ab_12_rsi5_above_value'),
            Integer(37, 60, name='sell_ab_13_rsi50_above_value'),

            Integer(-4, 10, name='sell_ab_14_roc_above_value'),#/2
            Integer(-2, 8, name='sell_ab_15_roc50_above_value'),
            Integer(-4, 8, name='sell_ab_16_roc2_above_value'),#/2

#--------------------------------

            ##CHANGE DEPENDING WHAT TYPE OF SELL  --> PEAK  OR DOWTRENDS
            Integer(-4, 6, name='sell_ab_17_ppo5_above_value'),#/2
            Integer(-4, 6, name='sell_ab_18_ppo10_above_value'),#/2
            Integer(-10, 8, name='sell_ab_19_ppo25_above_value'),#/2

            Integer(-10, 8, name='sell_ab_20_ppo50_above_value'),#/2
            Integer(-6, 6, name='sell_ab_21_ppo100_above_value'),
            Integer(-6, 6, name='sell_ab_22_ppo200_above_value'),
            Integer(-4, 5, name='sell_ab_23_ppo500_above_value'),#*2

            # ##USE AT A LATER STEP
            #
            # Integer(-1, 6, name='sell_ab_24_convsmall_above_value'),#/2 # extreme 12
            # Integer(-1, 4, name='sell_ab_25_convmedium_above_value'),# extreme 6
            # Integer(-1, 7, name='sell_ab_26_convlarge_above_value'),# extreme 12
            # Integer(-1, 8, name='sell_ab_27_convultra_above_value'),#/2# extreme 12
            #
            # Integer(-1, 6, name='sell_ab_28_convdist_above_value'), #very extreme not useful 10+

#-------------------------------------------------------------------------------------------------------

            #SMA'S GOING DOWN

            Categorical([True, False], name='sell_down_0a_sma3'),
            Categorical([True, False], name='sell_down_0b_sma5'),
            Categorical([True, False], name='sell_down_1_sma10'),
            Categorical([True, False], name='sell_down_2_sma25'),
            Categorical([True, False], name='sell_down_3_sma50'),
            Categorical([True, False], name='sell_down_4_sma100'),
            Categorical([True, False], name='sell_down_5_sma200'),

            Categorical([True, False], name='sell_down_6_sma400'),
            Categorical([True, False], name='sell_down_7_sma10k'),
            # Categorical([True, False], name='sell_down_8_sma20k'),
            # Categorical([True, False], name='sell_down_9_sma30k'),

            Categorical([True, False], name='sell_down_10_convsmall'),
            Categorical([True, False], name='sell_down_11_convmedium'),
            Categorical([True, False], name='sell_down_12_convlarge'),
            Categorical([True, False], name='sell_down_13_convultra'),
            Categorical([True, False], name='sell_down_14_convdist'),

            Categorical([True, False], name='sell_down_15_vol50'),
            Categorical([True, False], name='sell_down_16_vol100'),
            Categorical([True, False], name='sell_down_17_vol175'),
            Categorical([True, False], name='sell_down_18_vol250'),
            Categorical([True, False], name='sell_down_19_vol500'),

            Categorical([True, False], name='sell_down_20_vol1000'),
            Categorical([True, False], name='sell_down_21_vol100mean'),
            Categorical([True, False], name='sell_down_22_vol250mean'),

#-------------------------------------------------------------------------------------------------------

            ##ABOVE/BELOW SMAS

            Categorical([-1, 0, 1], name='sell_ab_1_sma10'),
            Categorical([-1, 0, 1], name='sell_ab_2_sma25'),
            Categorical([-1, 0, 1], name='sell_ab_3_sma50'),

            Categorical([-1, 0, 1], name='sell_ab_4_sma100'),
            Categorical([-1, 0, 1], name='sell_ab_5_sma200'),
            Categorical([-1, 0, 1], name='sell_ab_6_sma400'),
            Categorical([-1, 0, 1], name='sell_ab_7_sma10k'),

#-------------------------------------------------------------------------------------------------------

            ##DOWNSWINGS / UPSWINGS PPO'S

            ##UP OR DOWN (1 UP, 0 NOTHING, -1 DOWN)

            Categorical([-1, 0, 1], name='sell_swings_1_ppo5_up_or_down_bool'),
            Categorical([-1, 0, 1], name='sell_swings_2_ppo10_up_or_down_bool'),
            Categorical([-1, 0], name='sell_swings_3_ppo25_up_or_down_bool'),

            Categorical([-1, 0], name='sell_swings_4_ppo50_up_or_down_bool'),
            Categorical([-1, 0], name='sell_swings_5_ppo100_up_or_down_bool'),
            Categorical([-1, 0], name='sell_swings_6_ppo200_up_or_down_bool'),
            Categorical([-1, 0], name='sell_swings_7_ppo500_up_or_down_bool'),

            Categorical([-1, 0], name='sell_swings_8_roc50_up_or_down_bool'),
            Categorical([-1, 0], name='sell_swings_9_roc10_up_or_down_bool'),

#-------------------------------------------------------------------------------------------------------

            #DISTANCES

            #FOR MORE TOP SELLERS
            Integer(-6, 14, name='sell_dist_1_dist50_more_value'), #extreme, useless -4 ,30
            Integer(-8, 20, name='sell_dist_2_dist200_more_value'), #extreme, useless  -12-40
            Integer(-15, 30, name='sell_dist_3_dist400_more_value'),
            Integer(-15, 35, name='sell_dist_4_dist10k_more_value'),

            #FOR MORE TOP SELLERS
            Integer(-30, 25, name='sell_dist_5_dist_upbol50_more_value'),#/2
            Integer(-30, 25, name='sell_dist_6_dist_upbol100_more_value'),#/2


            #FOR MORE DOWNTREND BUYS LIKELY
            # Integer(-8, 50, name='sell_dist_7_dist_lowbol50_more_value'),#/2  ##set to more, as in higher from lower boll
            # Integer(-8, 50, name='sell_dist_8_dist_lowbol100_more_value'),#/2 ##set to more, as in higher from lower boll

            # Integer(-70, 40, name='sell_dist_7_roc50sma_more_value'),#*2  ##fix less more
            # Integer(-40, 12, name='sell_dist_8_roc200sma_more_value'),#*2

            ##below high 100
            #Integer(0, 0, name='sell_dist_9_high100_more_value'),

#-------------------------------------------------------------------------------------------------------




        ]



    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the sell strategy parameters to be used by hyperopt
        """
        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Sell strategy Hyperopt will build and use
            """
            # print(params)
            conditions = []
            # GUARDS AND TRENDS


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


            ##MAIN SELECTORS

#--------------------

            ##VOLATILITY

            conditions.append(dataframe['vol_mid']  > 0  )

           # conditions.append((dataframe['vol_low']  > 0) |(dataframe['vol_mid']  > 0) )

            # conditions.append((dataframe['vol_high']  > 0) |(dataframe['vol_mid']  > 0) )

#--------------------


            ##PICKS TREND COMBO

            conditions.append(

                (dataframe['uptrend'] >= params['main_1_trend_strength'])
                |#OR &
                (dataframe['uptrendsmall'] >= params['main_2_trend_strength'])

            )

            ##UPTREND
            #conditions.append(dataframe['uptrend'] >= params['main_1_trend_strength'])
            ##DOWNTREND
            #conditions.append(dataframe['downtrend'] >= params['main_1_trend_strength'])
            ##NOTREND
            #conditions.append((dataframe['uptrend'] <1)&(dataframe['downtrend'] <1))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            ##ABOVE/BELOW VALUES

            #RSI BELOW
            if 'include_ab_9_rsi_below_value' in params and params['include_ab_9_rsi_below_value']:
                conditions.append(dataframe['rsi'] < params['ab_9_rsi_below_value'])
            #RSI RECENT PEAK 5
            if 'include_ab_10_rsi_recent_peak_2_value' in params and params['include_ab_10_rsi_recent_peak_2_value']:
                conditions.append(dataframe['rsi'].rolling(2).max() < params['ab_10_rsi_recent_peak_2_value'])

            #RSI RECENT PEAK 12
            if 'include_ab_11_rsi_recent_peak_4_value' in params and params['include_ab_11_rsi_recent_peak_4_value']:
                conditions.append(dataframe['rsi'].rolling(4).max() < params['ab_11_rsi_recent_peak_4_value'])
            #RSI5 BELOW
            if 'include_ab_12_rsi5_below_value' in params and params['include_ab_12_rsi5_below_value']:
                conditions.append(dataframe['rsi5'] < params['ab_12_rsi5_below_value'])
            #RSI50 BELOW
            if 'include_ab_13_rsi50_below_value' in params and params['include_ab_13_rsi50_below_value']:
                conditions.append(dataframe['rsi50'] < params['ab_13_rsi50_below_value'])

#-----------------------

            #ROC BELOW
            if 'include_ab_14_roc_below_value' in params and params['include_ab_14_roc_below_value']:
                conditions.append(dataframe['roc'] < (params['ab_14_roc_below_value']/2))
            #ROC50 BELOW
            if 'include_ab_15_roc50_below_value' in params and params['include_ab_15_roc50_below_value']:
                conditions.append(dataframe['roc50'] < (params['ab_15_roc50_below_value']))
            #ROC2 BELOW
            if 'include_ab_16_roc2_below_value' in params and params['include_ab_16_roc2_below_value']:
                conditions.append(dataframe['roc2'] < (params['ab_16_roc2_below_value']/2))

#-----------------------

            #PPO5 BELOW
            if 'include_ab_17_ppo5_below_value' in params and params['include_ab_17_ppo5_below_value']:
                conditions.append(dataframe['ppo5'] < (params['ab_17_ppo5_below_value']/2))
            #PPO10 BELOW
            if 'include_ab_18_ppo10_below_value' in params and params['include_ab_18_ppo10_below_value']:
                conditions.append(dataframe['ppo10'] < (params['ab_18_ppo10_below_value']/2))
            #PPO25 BELOW
            if 'include_ab_19_ppo25_below_value' in params and params['include_ab_19_ppo25_below_value']:
                conditions.append(dataframe['ppo25'] < (params['ab_19_ppo25_below_value']/2))

            #PPO50 BELOW
            if 'include_ab_20_ppo50_below_value' in params and params['include_ab_20_ppo50_below_value']:
                conditions.append(dataframe['ppo50'] < (params['ab_20_ppo50_below_value']/2))
            #PPO100 BELOW
            if 'include_ab_21_ppo100_below_value' in params and params['include_ab_21_ppo100_below_value']:
                conditions.append(dataframe['ppo100'] < (params['ab_21_ppo100_below_value']))
            #PPO200 BELOW
            if 'include_ab_22_ppo200_below_value' in params and params['include_ab_22_ppo200_below_value']:
                conditions.append(dataframe['ppo200'] < (params['ab_22_ppo200_below_value']))
            #PPO500 BELOW
            if 'include_ab_23_ppo500_below_value' in params and params['include_ab_23_ppo500_below_value']:
                conditions.append(dataframe['ppo500'] < (params['ab_23_ppo500_below_value']*2))

            ##USE AT A LATER STEP

            #convsmall BELOW
            if 'include_ab_24_convsmall_below_value' in params and params['include_ab_24_convsmall_below_value']:
                conditions.append(dataframe['convsmall'] < (params['ab_24_convsmall_below_value']/2))
            #convmedium BELOW
            if 'include_ab_25_convmedium_below_value' in params and params['include_ab_25_convmedium_below_value']:
                conditions.append(dataframe['convmedium'] < (params['ab_25_convmedium_below_value']))
            #convlarge BELOW
            if 'include_ab_26_convlarge_below_value' in params and params['include_ab_26_convlarge_below_value']:
                conditions.append(dataframe['convlarge'] < (params['ab_26_convlarge_below_value']))
            #convultra BELOW
            if 'include_ab_27_convultra_below_value' in params and params['include_ab_27_convultra_below_value']:
                conditions.append(dataframe['convultra'] < (params['ab_27_convultra_below_value']/2))
            #convdist BELOW
            if 'include_ab_28_convdist_below_value' in params and params['include_ab_28_convdist_below_value']:
                conditions.append(dataframe['convdist'] < (params['ab_28_convdist_below_value']))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


            ##SMA'S GOING UP

            if 'up_0a_sma3' in params and params['up_0a_sma3']:
                conditions.append((dataframe['sma3'].shift(1) <dataframe['sma3']))
            if 'up_0b_sma5' in params and params['up_0b_sma5']:
                conditions.append((dataframe['sma5'].shift(1) <dataframe['sma5']))
            if 'up_1_sma10' in params and params['up_1_sma10']:
                conditions.append((dataframe['sma10'].shift(1) <dataframe['sma10']))
            if 'up_2_sma25' in params and params['up_2_sma25']:
                conditions.append((dataframe['sma25'].shift(1) <dataframe['sma25']))
            if 'up_3_sma50' in params and params['up_3_sma50']:
                conditions.append((dataframe['sma50'].shift(2) <dataframe['sma50']))
            if 'up_4_sma100' in params and params['up_4_sma100']:
                conditions.append((dataframe['sma100'].shift(3) <dataframe['sma100']))
            if 'up_5_sma200' in params and params['up_5_sma200']:
                conditions.append((dataframe['sma200'].shift(4) <dataframe['sma200']))

            if 'up_6_sma400' in params and params['up_6_sma400']:
                conditions.append((dataframe['sma400'].shift(4) <dataframe['sma400']))
            if 'up_7_sma10k' in params and params['up_7_sma10k']:
                conditions.append((dataframe['sma10k'].shift(5) <dataframe['sma10k']))
            # if 'up_8_sma20k' in params and params['up_8_sma20k']:
            #     conditions.append((dataframe['sma20k'].shift(5) <dataframe['sma20k']))
            # if 'up_9_sma30k' in params and params['up_9_sma30k']:
            #     conditions.append((dataframe['sma30k'].shift(5) <dataframe['sma30k']))

            if 'up_10_convsmall' in params and params['up_10_convsmall']:
                conditions.append((dataframe['convsmall'].shift(2) <dataframe['convsmall']))
            if 'up_11_convmedium' in params and params['up_11_convmedium']:
                conditions.append((dataframe['convmedium'].shift(3) <dataframe['convmedium']))
            if 'up_12_convlarge' in params and params['up_12_convlarge']:
                conditions.append((dataframe['convlarge'].shift(4) <dataframe['convlarge']))
            if 'up_13_convultra' in params and params['up_13_convultra']:
                conditions.append((dataframe['convultra'].shift(4) <dataframe['convultra']))
            if 'up_14_convdist' in params and params['up_14_convdist']:
                conditions.append((dataframe['convdist'].shift(4) <dataframe['convdist']))

            if 'up_15_vol50' in params and params['up_15_vol50']:
                conditions.append((dataframe['vol50'].shift(2) <dataframe['vol50']))
            if 'up_16_vol100' in params and params['up_16_vol100']:
                conditions.append((dataframe['vol100'].shift(3) <dataframe['vol100']))
            if 'up_17_vol175' in params and params['up_17_vol175']:
                conditions.append((dataframe['vol175'].shift(4) <dataframe['vol175']))
            if 'up_18_vol250' in params and params['up_18_vol250']:
                conditions.append((dataframe['vol250'].shift(4) <dataframe['vol250']))
            if 'up_19_vol500' in params and params['up_19_vol500']:
                conditions.append((dataframe['vol500'].shift(4) <dataframe['vol500']))

            if 'up_20_vol1000' in params and params['up_20_vol1000']:
                conditions.append((dataframe['vol1000'].shift(4) <dataframe['vol1000']))
            if 'up_21_vol100mean' in params and params['up_21_vol100mean']:
                conditions.append((dataframe['vol100mean'].shift(4) <dataframe['vol100mean']))
            if 'up_22_vol250mean' in params and params['up_22_vol250mean']:
                conditions.append((dataframe['vol250mean'].shift(4) <dataframe['vol250mean']))


            if 'up_20_conv3' in params and params['up_20_conv3']:
                conditions.append(((dataframe['conv3'].shift(25) < dataframe['conv3'])&(dataframe['conv3'].shift(50) < dataframe['conv3'])))
            if 'up_21_vol5' in params and params['up_21_vol5']:
                conditions.append(((dataframe['vol5'].shift(25) < dataframe['vol5'])&(dataframe['vol5'].shift(50) < dataframe['vol5'])))
            if 'up_22_vol5ultra' in params and params['up_22_vol5ultra']:
                conditions.append(((dataframe['vol5ultra'].shift(25) < dataframe['vol5ultra'])&(dataframe['vol5ultra'].shift(50) < dataframe['vol5ultra'])))
            if 'up_23_vol1ultra' in params and params['up_23_vol1ultra']:
                conditions.append(((dataframe['vol1ultra'].shift(25) < dataframe['vol1ultra'])& (dataframe['vol1ultra'].shift(50) < dataframe['vol1ultra'])))
            if 'up_24_vol1' in params and params['up_24_vol1']:
                conditions.append(((dataframe['vol1'].shift(30) < dataframe['vol1'])&(dataframe['vol1'].shift(10) < dataframe['vol1'])))
            if 'up_25_vol5inc24' in params and params['up_25_vol5inc24']:
                conditions.append((dataframe['vol5inc24'].shift(50) < dataframe['vol5inc24']))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            ##ABOVE / BELOW SMAS    1 above/ 0 None  / -1 below

            #SMA10
            conditions.append((dataframe['close'] > dataframe['sma10'])|(0.5 > params['ab_1_sma10']))
            conditions.append((dataframe['close'] < dataframe['sma10'])|(-0.5 < params['ab_1_sma10']))
            #SMA25
            conditions.append((dataframe['close'] > dataframe['sma25'])|(0.5 > params['ab_2_sma25']))
            conditions.append((dataframe['close'] < dataframe['sma25'])|(-0.5 < params['ab_2_sma25']))
            #SMA50
            conditions.append((dataframe['close'] > dataframe['sma50'])|(0.5 > params['ab_3_sma50']))
            conditions.append((dataframe['close'] < dataframe['sma50'])|(-0.5 < params['ab_3_sma50']))


            #SMA100
            conditions.append((dataframe['close'] > dataframe['sma100'])|(0.5 > params['ab_4_sma100']))
            conditions.append((dataframe['close'] < dataframe['sma100'])|(-0.5 < params['ab_4_sma100']))
            #SMA100
            conditions.append((dataframe['close'] > dataframe['sma200'])|(0.5 > params['ab_5_sma200']))
            conditions.append((dataframe['close'] < dataframe['sma200'])|(-0.5 < params['ab_5_sma200']))
            #SMA400
            conditions.append((dataframe['close'] > dataframe['sma400'])|(0.5 > params['ab_6_sma400']))
            conditions.append((dataframe['close'] < dataframe['sma400'])|(-0.5 < params['ab_6_sma400']))
            #SMA10k
            conditions.append((dataframe['close'] > dataframe['sma10k'])|(0.5 > params['ab_7_sma10k']))
            conditions.append((dataframe['close'] < dataframe['sma10k'])|(-0.5 < params['ab_7_sma10k']))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            ##DOWNSWINGS / UPSWINGS PPO'S

            #ppo5  UP OR DOWN (1 UP, 0 NOTHING, -1 DOWN)
            conditions.append((dataframe['ppo5'].shift(1) <dataframe['ppo5'])|(0.5 > params['swings_1_ppo5_up_or_down_bool']))
            conditions.append((dataframe['ppo5'].shift(1) >dataframe['ppo5'])|(-0.5 < params['swings_1_ppo5_up_or_down_bool']))
            #ppo10
            conditions.append((dataframe['ppo10'].shift(1) <dataframe['ppo10'])|(0.5 > params['swings_2_ppo10_up_or_down_bool']))
            conditions.append((dataframe['ppo10'].shift(1) >dataframe['ppo10'])|(-0.5 < params['swings_2_ppo10_up_or_down_bool']))
            #ppo25
            conditions.append((dataframe['ppo25'].shift(1) <dataframe['ppo25'])|(0.5 > params['swings_3_ppo25_up_or_down_bool']))
            #conditions.append((dataframe['ppo25'].shift(1) >dataframe['ppo25'])|(-0.5 < params['swings_3_ppo25_up_or_down_bool']))

            #ppo50
            conditions.append((dataframe['ppo50'].shift(2) <dataframe['ppo50'])|(0.5 > params['swings_4_ppo50_up_or_down_bool']))
            #conditions.append((dataframe['ppo50'].shift(2) >dataframe['ppo50'])|(-0.5 < params['swings_4_ppo50_up_or_down_bool']))
            #ppo100
            conditions.append((dataframe['ppo100'].shift(3) <dataframe['ppo100'])|(0.5 > params['swings_5_ppo100_up_or_down_bool']))
            #conditions.append((dataframe['ppo100'].shift(3) >dataframe['ppo100'])|(-0.5 < params['swings_5_ppo100_up_or_down_bool']))
            #ppo200
            conditions.append((dataframe['ppo200'].shift(4) <dataframe['ppo200'])|(0.5 > params['swings_6_ppo200_up_or_down_bool']))
            #conditions.append((dataframe['ppo200'].shift(4) >dataframe['ppo200'])|(-0.5 < params['swings_6_ppo200_up_or_down_bool']))
            #ppo500
            conditions.append((dataframe['ppo500'].shift(5) <dataframe['ppo500'])|(0.5 > params['swings_7_ppo500_up_or_down_bool']))
            #conditions.append((dataframe['ppo500'].shift(5) >dataframe['ppo500'])|(-0.5 < params['swings_7_ppo500_up_or_down_bool']))

            #roc50
            conditions.append((dataframe['roc50'].shift(2) <dataframe['roc50'])|(0.5 > params['swings_8_roc50_up_or_down_bool']))
            #conditions.append((dataframe['roc50'].shift(3) >dataframe['roc50'])|(-0.5 < params['swings_8_roc50_up_or_down_bool']))
            #roc10
            conditions.append((dataframe['roc10'].shift(1) <dataframe['roc10'])|(0.5 > params['swings_9_roc10_up_or_down_bool']))
            #conditions.append((dataframe['roc10'].shift(2) >dataframe['roc10'])|(-0.5 < params['swings_9_roc10_up_or_down_bool']))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


            ##DISTANCES/ROC

            #dist50 LESS THAN
            if 'include_dist_1_dist50_less_value' in params and params['include_dist_1_dist50_less_value']:
                conditions.append(dataframe['dist50'] < (params['dist_1_dist50_less_value']))
            #dist200 LESS THAN
            if 'include_dist_2_dist200_less_value' in params and params['include_dist_2_dist200_less_value']:
                conditions.append(dataframe['dist200'] < (params['dist_2_dist200_less_value']))

            #dist400 LESS THAN
            if 'include_dist_3_dist400_less_value' in params and params['include_dist_3_dist400_less_value']:
                conditions.append(dataframe['dist400'] < (params['dist_3_dist400_less_value']))
            #dist10k LESS THAN
            if 'include_dist_4_dist10k_less_value' in params and params['include_dist_4_dist10k_less_value']:
                conditions.append(dataframe['dist10k'] < (params['dist_4_dist10k_less_value']))

            #less =further from top bol
            #dist_upbol50 LESS THAN
            if 'include_dist_5_dist_upbol50_less_value' in params and params['include_dist_5_dist_upbol50_less_value']:
                conditions.append(dataframe['dist_upbol50'] < (params['dist_5_dist_upbol50_less_value']/2))
            #dist_upbol100 LESS THAN
            if 'include_dist_6_dist_upbol100_less_value' in params and params['include_dist_6_dist_upbol100_less_value']:
                conditions.append(dataframe['dist_upbol100'] < (params['dist_6_dist_upbol100_less_value']/2))

            # #less =closer to bot bol
            # #dist_upbol50 LESS THAN
           # if 'include_dist_7_dist_lowbol50_less_value' in params and params['include_dist_7_dist_lowbol50_less_value']:
            #   conditions.append(dataframe['dist_lowbol50'] < (params['dist_7_dist_lowbol50_less_value']/2))
            # #dist_upbol100 LESS THAN
           # if 'include_dist_8_dist_lowbol100_less_value' in params and params['include_dist_8_dist_lowbol100_less_value']:
            #   conditions.append(dataframe['dist_lowbol100'] < (params['dist_8_dist_lowbol100_less_value']/2))



            #others
            ##roc50sma MORE THAN
            if 'include_dist_7_roc50sma_less_value' in params and params['include_dist_7_roc50sma_less_value']:
                conditions.append(dataframe['roc50sma'] < (params['dist_7_roc50sma_less_value']*2))
            #roc200sma MORE THAN
            if 'include_dist_8_roc200sma_less_value' in params and params['include_dist_8_roc200sma_less_value']:
                conditions.append(dataframe['roc200sma'] < (params['dist_8_roc200sma_less_value']*2))

            ##ENABLE TO BUY AWAY FROM HIGH
            # #HIGH500 TO CLOSE  MORE THAN
            #if 'include_dist_9_high100_more_value' in params and params['include_dist_9_high100_more_value']:
            #   conditions.append((dataframe['high100']-dataframe['close']) > ((dataframe['high100']/100* (params['dist_9_high100_more_value']))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





            # Check that volume is not 0
            conditions.append(dataframe['volume'] > 0)

            if conditions:


                #  ##ENABLE SELLS ALWAYS ON OTHER VOLATILITYS
                # dataframe.loc[
                #     ((dataframe['vol_low']  > 0) |(dataframe['vol_high']  > 0) ),
                #     'sell'] = 1


                # ##ENABLE PRODUCTION SELLS
                # dataframe.loc[
                #     (add_production_sells_low(dataframe)),
                #     'sell'] = 1
                #

                dataframe.loc[
                    (~(reduce(lambda x, y: x & y, conditions)))&OPTIMIZED_RULE(dataframe,params),
                    'sell'] = 1

            return dataframe

        return populate_sell_trend

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching sell strategy parameters
        """
        return [


#-------------------------------------------------------------------------------------------------------

            ## CUSTOM RULE TRESHOLDS

            # SKDecimal(5.0, 7.0,decimals=1, name='sell_trigger_0_roc_ab_value'),# 5       range 5-7  or 4-7
            # SKDecimal(3.2, 4.5,decimals=1, name='sell_trigger_0_roc2_ab_value'),#3.8   range 3.2-4.5
            # Integer(77, 82, name='sell_trigger_0_rsi_ab_value'),#79      range 82-77
            # Integer(90, 95, name='sell_trigger_0_rsi5_ab_value'),#94    range  95-90
            # Integer(63, 67, name='sell_trigger_0_rsi50_ab_value'),#66   range 67-63

#-------------------------------------------------------------------------------------------------------

            ##MAIN

            Categorical([1, 2, 3], name='main_1_trend_strength'), #UPTREND STR
            Categorical([1, 2, 3], name='main_2_trend_strength'), #SMALL UPTREND STR


            #Categorical([-1, 0, 1], name='main_2_small_uptrend_downtrend'), #SMALL UPTREND ON/OFF  1 is on -1 is down

#-------------------------------------------------------------------------------------------------------

            ##INCLUDE/EXCLUDE RULES

            Categorical([True, False], name='include_ab_9_rsi_below_value'),
            Categorical([True, False], name='include_ab_10_rsi_recent_peak_2_value'),
            Categorical([True, False], name='include_ab_11_rsi_recent_peak_4_value'),
            Categorical([True, False], name='include_ab_12_rsi5_below_value'),
            Categorical([True, False], name='include_ab_13_rsi50_below_value'),

            Categorical([True, False], name='include_ab_14_roc_below_value'),
            Categorical([True, False], name='include_ab_15_roc50_below_value'),
            Categorical([True, False], name='include_ab_16_roc2_below_value'),

            Categorical([True, False], name='include_ab_17_ppo5_below_value'),
            Categorical([True, False], name='include_ab_18_ppo10_below_value'),
            Categorical([True, False], name='include_ab_19_ppo25_below_value'),

            Categorical([True, False], name='include_ab_20_ppo50_below_value'),
            Categorical([True, False], name='include_ab_21_ppo100_below_value'),
            Categorical([True, False], name='include_ab_22_ppo200_below_value'),
            Categorical([True, False], name='include_ab_23_ppo500_below_value'),

            ##USE AT A LATER STEP
            Categorical([True, False], name='include_ab_24_convsmall_below_value'),
            Categorical([True, False], name='include_ab_25_convmedium_below_value'),
            Categorical([True, False], name='include_ab_26_convlarge_below_value'),
            Categorical([True, False], name='include_ab_27_convultra_below_value'),

            Categorical([True, False], name='include_ab_28_convdist_below_value'),

            Categorical([True, False], name='include_dist_1_dist50_less_value'),
            Categorical([True, False], name='include_dist_2_dist200_less_value'),
            Categorical([True, False], name='include_dist_3_dist400_less_value'),
            Categorical([True, False], name='include_dist_4_dist10k_less_value'),

            Categorical([True, False], name='include_dist_5_dist_upbol50_less_value'),
            Categorical([True, False], name='include_dist_6_dist_upbol100_less_value'),


            # FOR MORE DOWNTREND BUYS LIKELY
            # Categorical([True, False],  name='include_dist_7_dist_lowbol50_less_value'),
            # Categorical([True, False],  name='include_dist_8_dist_lowbol100_less_value'),

            #MORE LIKE TRIGGERS
            Categorical([True, False],  name='include_dist_7_roc50sma_less_value'),
            Categorical([True, False],  name='include_dist_8_roc200sma_less_value'),

            ##below high 100
            #Categorical([True, False],  name='include_dist_9_high100_more_value'),



#-------------------------------------------------------------------------------------------------------

            ##ABOVE/BELOW VALUES

            Integer(35, 75, name='ab_9_rsi_below_value'),
            Integer(60, 82, name='ab_10_rsi_recent_peak_2_value'),
            Integer(60, 82, name='ab_11_rsi_recent_peak_4_value'),
            Integer(40, 101, name='ab_12_rsi5_below_value'),
            Integer(37, 73, name='ab_13_rsi50_below_value'),

            Integer(-6, 10, name='ab_14_roc_below_value'),#/2
            Integer(-8, 8, name='ab_15_roc50_below_value'),
            Integer(-4, 6, name='ab_16_roc2_below_value'),#/2

#--------------------------------

            Integer(-4, 4, name='ab_17_ppo5_below_value'),#/2
            Integer(-5, 5, name='ab_18_ppo10_below_value'),#/2
            Integer(-8, 10, name='ab_19_ppo25_below_value'),#/2

            Integer(-6, 7, name='ab_20_ppo50_below_value'),#/2
            Integer(-6, 7, name='ab_21_ppo100_below_value'),
            Integer(-5, 7, name='ab_22_ppo200_below_value'),
            Integer(-4, 4, name='ab_23_ppo500_below_value'),#*2

            ##USE AT A LATER STEP

            Integer(1, 12, name='ab_24_convsmall_below_value'),#/2 #final
            Integer(1, 6, name='ab_25_convmedium_below_value'),#final
            Integer(1, 15, name='ab_26_convlarge_below_value'), #final
            Integer(2, 12, name='ab_27_convultra_below_value'),#/2 #final

            Integer(2, 30, name='ab_28_convdist_below_value'),

#-------------------------------------------------------------------------------------------------------

             #SMA'S GOING UP

            Categorical([True, False], name='up_0a_sma3'),
            Categorical([True, False], name='up_0b_sma5'),
            Categorical([True, False], name='up_1_sma10'),
            Categorical([True, False], name='up_2_sma25'),
            Categorical([True, False], name='up_3_sma50'),
            Categorical([True, False], name='up_4_sma100'),
            Categorical([True, False], name='up_5_sma200'),

            Categorical([True, False], name='up_6_sma400'),
            Categorical([True, False], name='up_7_sma10k'),
            # Categorical([True, False], name='up_8_sma20k'),
            # Categorical([True, False], name='up_9_sma30k'),

            Categorical([True, False], name='up_10_convsmall'),
            Categorical([True, False], name='up_11_convmedium'),
            Categorical([True, False], name='up_12_convlarge'),
            Categorical([True, False], name='up_13_convultra'),
            Categorical([True, False], name='up_14_convdist'),

            Categorical([True, False], name='up_15_vol50'),
            Categorical([True, False], name='up_16_vol100'),
            Categorical([True, False], name='up_17_vol175'),
            Categorical([True, False], name='up_18_vol250'),
            Categorical([True, False], name='up_19_vol500'),

            Categorical([True, False], name='up_20_vol1000'),
            Categorical([True, False], name='up_21_vol100mean'),
            Categorical([True, False], name='up_22_vol250mean'),

#-------------------------------------------------------------------------------------------------------

            ##ABOVE/BELOW SMAS

            Categorical([-1, 0, 1], name='ab_1_sma10'),
            Categorical([-1, 0, 1], name='ab_2_sma25'),
            Categorical([-1, 0, 1], name='ab_3_sma50'),

            Categorical([-1, 0, 1], name='ab_4_sma100'),
            Categorical([-1, 0, 1], name='ab_5_sma200'),
            Categorical([-1, 0, 1], name='ab_6_sma400'),
            Categorical([-1, 0, 1], name='ab_7_sma10k'),

#-------------------------------------------------------------------------------------------------------

            ##DOWNSWINGS / UPSWINGS PPO'S

            ##UP OR DOWN (1 UP, 0 NOTHING, -1 DOWN)

            Categorical([-1, 0, 1],  name='swings_1_ppo5_up_or_down_bool'), # -1 down, 1 up , 0 off
            Categorical([-1, 0, 1],name='swings_2_ppo10_up_or_down_bool'),
            Categorical([-1, 0, 1], name='swings_3_ppo25_up_or_down_bool'),   #1 up , 0 off

            Categorical([0, 1], name='swings_4_ppo50_up_or_down_bool'),
            Categorical([0, 1], name='swings_5_ppo100_up_or_down_bool'),
            Categorical([0, 1], name='swings_6_ppo200_up_or_down_bool'),
            Categorical([ 0, 1],name='swings_7_ppo500_up_or_down_bool'),

            Categorical([0, 1], name='swings_8_roc50_up_or_down_bool'),
            Categorical([0, 1], name='swings_9_roc10_up_or_down_bool'),

#-------------------------------------------------------------------------------------------------------

            ##DISTANCES

            Integer(-7, 14, name='dist_1_dist50_less_value'), ##extreme 8-30
            Integer(-8, 25, name='dist_2_dist200_less_value'), ##extreme 12 -40
            Integer(-12, 35, name='dist_3_dist400_less_value'),
            Integer(-12, 40, name='dist_4_dist10k_less_value'),

            Integer(-25, 30, name='dist_5_dist_upbol50_less_value'),#/2
            Integer(-25, 30, name='dist_6_dist_upbol100_less_value'),#/2


            # FOR MORE DOWNTREND BUYS LIKELY
            # Integer(-6, 100, name='dist_7_dist_lowbol50_less_value'),#/2
            # Integer(-6, 100, name='dist_8_dist_lowbol100_less_value'),#/2

            ##MORE LIKE TRIGGERS
            # Integer(-40, 70, name='dist_7_roc50sma_less_value'),#*2 ##pretty extreme
            # Integer(-12, 40, name='dist_8_roc200sma_less_value'),#*2

            ##below high 100
            #Integer(0, 0, name='dist_9_high100_more_value'),

#-------------------------------------------------------------------------------------------------------





        ]


def OPTIMIZED_RULE(dataframe,params):
    return(

                (dataframe['sma100'] < dataframe['close'])

        )

def add_production_buys_mid(dataframe):
    return(

        MID_VOLATILITY(dataframe)
             &
        mid_volatility_buyer(dataframe)
    )

def add_production_sells_mid(dataframe):
    return(

        MID_VOLATILITY(dataframe)
             &
        mid_volatility_seller(dataframe)
    )


