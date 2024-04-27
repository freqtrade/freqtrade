FROM freqtradeorg/freqtrade:develop_freqairl

USER root
# Install dependencies
COPY requirements-dev.txt /freqtrade/

ARG USERNAME=ftuser

RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog \
    && apt-get -y install --no-install-recommends git sudo ssh vim build-essential \
    && apt-get clean \
    && mkdir -p /home/${USERNAME}/.vscode-server /home/${USERNAME}/.vscode-server-insiders /home/${USERNAME}/commandhistory \
    # Add user to Suders
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # keep command history
    && echo "export PROMPT_COMMAND='history -a'" >> /home/${USERNAME}/.bashrc \
    && echo "export HISTFILE=~/commandhistory/.bash_history" >> /home/${USERNAME}/.bashrc \
    && chown ${USERNAME}:${USERNAME} -R /home/${USERNAME}/.local/ \
    && chown ${USERNAME}: -R /home/${USERNAME}/

USER ftuser

RUN pip install --user autopep8 -r docs/requirements-docs.txt -r requirements-dev.txt --no-cache-dir

# Empty the ENTRYPOINT to allow all commands
ENTRYPOINT []
