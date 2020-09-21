FROM freqtradeorg/freqtrade:develop

# Install dependencies
COPY requirements-dev.txt /freqtrade/
RUN apt-get update \
    && apt-get -y install git sudo vim \
    && apt-get clean \
    && pip install autopep8 -r docs/requirements-docs.txt -r requirements-dev.txt --no-cache-dir \
    && useradd -u 1000 -U -m ftuser \
    && mkdir -p /home/ftuser/.vscode-server /home/ftuser/.vscode-server-insiders /home/ftuser/commandhistory \
    && echo "export PROMPT_COMMAND='history -a'" >> /home/ftuser/.bashrc \
    && echo "export HISTFILE=~/commandhistory/.bash_history" >> /home/ftuser/.bashrc \
    && chown ftuser: -R /home/ftuser/

USER ftuser

# Empty the ENTRYPOINT to allow all commands
ENTRYPOINT []
