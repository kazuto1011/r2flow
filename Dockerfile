FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# ====================================================
# foudamentals
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential curl fzf git htop sudo tmux tree vim wget zsh \
    ninja-build libsparsehash-dev ffmpeg
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm --recursive --force /var/lib/apt/lists/*

ARG UNAME=user
ARG UID=1000
ARG GID=$UID

RUN groupadd --gid $GID $UNAME \
    && useradd -m --uid $UID --gid $GID -G sudo $UNAME
RUN echo "$UNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$UNAME \
    && chmod 440 /etc/sudoers.d/$UNAME

USER $UNAME
ARG HOME="/home/$UNAME"

# ====================================================
# python env
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR $HOME
RUN uv venv --python 3.10
ENV UV_PROJECT_ENVIRONMENT="$HOME/.venv/bin"
ENV PATH="$UV_PROJECT_ENVIRONMENT:$PATH"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0"
ENV FORCE_CUDA="1"

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt
RUN uv pip install git+https://github.com/mit-han-lab/torchsparse.git@v2.0.0 --no-build-isolation
RUN uv pip install natten==0.17.1+torch210cu121 --find-links https://shi-labs.com/natten/wheels/ --no-build-isolation

# ====================================================
WORKDIR $HOME/workspace