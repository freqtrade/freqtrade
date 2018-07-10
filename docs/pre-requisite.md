# Pre-requisite
Before running your bot in production you will need to setup few
external API. In production mode, the bot required valid Bittrex API
credentials and a Telegram bot (optional but recommended).

## Table of Contents
- [Setup your Bittrex account](#setup-your-bittrex-account)
- [Backtesting commands](#setup-your-telegram-bot)

## Setup your Bittrex account
*To be completed, please feel free to complete this section.*

## Setup your Telegram bot
The only things you need is a working Telegram bot and its API token.
Below we explain how to create your Telegram Bot, and how to get your
Telegram user id.

### 1. Create your Telegram bot
**1.1. Start a chat with https://telegram.me/BotFather**  
**1.2. Send the message** `/newbot`  
*BotFather response:*
```
Alright, a new bot. How are we going to call it? Please choose a name for your bot.
```
**1.3. Choose the public name of your bot (e.g "`Freqtrade bot`")**  
*BotFather response:*
```
Good. Now let's choose a username for your bot. It must end in `bot`. Like this, for example: TetrisBot or tetris_bot.
```
**1.4. Choose the name id of your bot (e.g "`My_own_freqtrade_bot`")**  
**1.5. Father bot will return you the token (API key)**  
Copy it and keep it you will use it for the config parameter `token`.  
*BotFather response:*
```
Done! Congratulations on your new bot. You will find it at t.me/My_own_freqtrade_bot. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.

Use this token to access the HTTP API:
521095879:AAEcEZEL7ADJ56FtG_qD0bQJSKETbXCBCi0

For a description of the Bot API, see this page: https://core.telegram.org/bots/api
```
**1.6. Don't forget to start the conversation with your bot, by clicking /START button**  

### 2. Get your user id
**2.1. Talk to https://telegram.me/userinfobot**  
**2.2. Get your "Id", you will use it for the config parameter 
`chat_id`.**

