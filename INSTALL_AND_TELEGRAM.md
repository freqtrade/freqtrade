# Install Freqtrade (AI version) and connect Telegram

Follow these steps in order. You already have the repo at `c:\Users\lis_8\ft_workspace\freqtrade`.

---

## 1. Install the bot with the AI version (FreqAI)

Open **PowerShell** in the project folder and run the official setup script:

```powershell
cd c:\Users\lis_8\ft_workspace\freqtrade
.\setup.ps1
```

When asked **"Select which requirement files to install"**, choose:

- **D** – `requirements-freqai.txt` (this is the AI version)

You can also add **A** (base `requirements.txt`) if not already included. Optionally add **C** (`requirements-hyperopt.txt`) for hyperopt. Separate multiple choices with commas, e.g. `A,D`.

The script will:

- Create a virtual environment (`.venv`)
- Install dependencies including FreqAI (scikit-learn, LightGBM, XGBoost, etc.)
- Install Freqtrade in editable mode
- Install FreqUI

If you prefer to install **without** the interactive script:

```powershell
cd c:\Users\lis_8\ft_workspace\freqtrade
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-freqai.txt
pip install -e .
freqtrade install-ui
```

---

## 2. Create user data directory

With the virtual environment activated:

```powershell
freqtrade create-userdir --userdir user_data
```

This creates `user_data` with subfolders: `strategies`, `notebooks`, `data`, `backtest_results`, etc., and copies sample strategy files.

---

## 3. Get your Telegram bot token and chat ID

1. **Create a bot and get token**
   - Open Telegram and search for **@BotFather**.
   - Send `/newbot`, set name and username (must end in `bot`).
   - Copy the **token** (e.g. `123456789:ABCdefGHI...`).

2. **Get your chat ID**
   - Search for **@userinfobot** in Telegram and start a chat.
   - It will reply with your **Id** (e.g. `123456789`). That is your `chat_id`.

3. **Start a chat with your bot**
   - Open your new bot (link from BotFather) and press **Start**.

---

## 4. Configure Telegram in Freqtrade

Edit `user_data\config.json` (created by `new-config` or use the example below).

Set under `"telegram"`:

- `"enabled": true`
- `"token": "YOUR_BOT_TOKEN_FROM_BOTFATHER"`
- `"chat_id": "YOUR_CHAT_ID_FROM_USERINFOBOT"`

Example:

```json
"telegram": {
    "enabled": true,
    "token": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
    "chat_id": "123456789"
}
```

If you don’t have a config yet, create one:

```powershell
freqtrade new-config --config user_data/config.json
```

Answer the prompts (exchange, dry-run, etc.). When asked **“Do you want to enable Telegram?”** choose **Yes** and enter your token and chat_id when asked.

Alternatively, a ready-made config with **Telegram and FreqAI enabled** is in `user_data\config_telegram_example.json`. Copy it to `user_data\config.json`, then edit and set:

- `telegram.token` – your bot token from BotFather  
- `telegram.chat_id` – your user id from userinfobot  
- `exchange.key` and `exchange.secret` – your exchange API keys (for live/dry-run with exchange)

---

## 5. Run the bot

Activate the venv and start trading (dry-run by default):

```powershell
cd c:\Users\lis_8\ft_workspace\freqtrade
.\.venv\Scripts\Activate.ps1
freqtrade trade --config user_data/config.json
```

With Telegram enabled you can control the bot from Telegram, for example:

- `/start` – start the trader  
- `/stop` – stop the trader  
- `/status` – open trades  
- `/balance` – balance  
- `/help` – list commands  

Full list: [Telegram usage](https://www.freqtrade.io/en/stable/telegram-usage/).

---

## 6. (Optional) Use an AI strategy (FreqAI)

To use FreqAI you need:

- In `config.json`: a `"freqai": { ... }` block with `"enabled": true` (see `config_examples/config_freqai.example.json`).
- A strategy that uses the FreqAI prediction interface (e.g. from the [FreqAI docs](https://www.freqtrade.io/en/stable/freqai/) or `user_data/strategies` examples).

The config example `config_examples/config_freqai.example.json` shows a full FreqAI setup (e.g. feature parameters, train period). You can merge that with your main config and Telegram section.

---

## Troubleshooting

- **“freqtrade: command not found”**  
  Activate the venv:  
  `.\\.venv\Scripts\Activate.ps1`

- **ExecutionPolicy error when running `.\setup.ps1`**  
  Run:  
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

- **Telegram bot doesn’t answer**  
  Ensure `telegram.enabled` is `true`, token and chat_id are correct, and you have started a chat with the bot (pressed Start).

- **C++ build errors on Windows**  
  Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (“Desktop development with C++”). Alternatively use WSL or Docker.
