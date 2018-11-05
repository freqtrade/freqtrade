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

Win rate means over X trades what is the perctange of winning trades to total number of trades (note that we don't consider how much you gained but only If you won or not).

<img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />