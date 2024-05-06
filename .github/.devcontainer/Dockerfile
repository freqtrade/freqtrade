FROM freqtradeorg/freqtrade:develop_freqairl

USER root
# Install dependencies
COPY requirements-dev.txt /freqtrade/

ARG USERNAME=ftuser

RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog git ssh vim build-essential zsh \
    && apt-get clean \
    && mkdir -p /home/${USERNAME}/.vscode-server /home/${USERNAME}/.vscode-server-insiders /home/${USERNAME}/commandhistory \
    && chown ${USERNAME}:${USERNAME} -R /home/${USERNAME}/.local/ \
    && chown ${USERNAME}: -R /home/${USERNAME}/

USER ftuser

RUN pip install --user autopep8 -r docs/requirements-docs.txt -r requirements-dev.txt --no-cache-dir

# Empty the ENTRYPOINT to allow all commands
ENTRYPOINT []
