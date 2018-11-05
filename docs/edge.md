#  Edge positioning

This page explains how to use Edge Positioning module in your bot in order to enter into a trade only of the trade has a reasonable win rate and risk reward ration, and consequently adjust your position size and stoploss.

## Table of Contents

- [Introduction](#introduction)

## Introduction
Trading is all about probability. no one can claim that he has the strategy working all the time. you have to assume that sometimes you lose.<br/><br/>
But it doesn't mean there is no rule, it only means rules should work "most of the time". let's play a game: we toss a coin, heads: I give you 10$, tails: You give me 10$. is it an interetsing game ? no, it is quite boring, isn't it?<br/><br/>
But lets say the probabiliy that we have heads is 80%, and the probablilty that we have tails is 20%. now it is becoming interesting ...
That means 10$ x 80% versus 10$ x 20%. 8$ versus 2$. that means over time you will win 8$ risking only 2$ on each toss of coin.<br/><br/>
lets complicate it more: you win 80% of the time but only 2$, I win 20% of the time but 8$. the calculation is: 80% * 2$ versus 20% * 8$. it is becoming boring again because overtime you win $1.6$ (80% x 2$) and me $1.6 (20% * 8$) too.<br/><br/>
The question is: how do you calculate that? how do you know if you wanna play?
The answer comes to two factors:
- Win Rate
- Risk Reward Ratio


### Win Rate
Means over X trades what is the perctange of winning trades to total number of trades (note that we don't consider how much you gained but only If you won or not).


W = (Number of winning trades) / (Number of losing trades)

### Risk Reward Ratio
Risk Reward Ratio is a formula used to measure the expected gains of a given investment against the risk of loss. it is basically what you potentially win divided by what you potentially lose:

R = Profit / Loss

Over time, on many trades, you can calculate your risk reward by dividing your average profit on winning trades by your average loss on losing trades:

average profit = (Sum of profits) / (Number of winning trades)

average loss = (Sum of losses) / (Number of losing trades)

R = (average profit) / (average loss)

### Expectancy

At this point we can combine W and R to create an expectancy ratio. This is a simple process of multiplying the risk reward ratio by the percentage of winning trades, and subtracting the percentage of losing trades, which is calculated as follows:

Expectancy Ratio = (Risk Reward Ratio x Win Rate) – Loss Rate

So lets say your Win rate is 28% and your Risk Reward Ratio is 5:

Expectancy = (5 * 0.28) - 0.72 = 0.68

Superficially, this means that on average you expect this strategy’s trades to return .68 times the size of your losers. This is important for two reasons: First, it may seem obvious, but you know right away that you have a positive return. Second, you now have a number you can compare to other candidate systems to make decisions about which ones you employ.

It is important to remember that any system with an expectancy greater than 0 is profitable using past data. The key is finding one that will be profitable in the future.

You can also use this number to evaluate the effectiveness of modifications to this system.
