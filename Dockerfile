FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# ====================================================
# fundamentals
# ====================================================

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential curl fzf git htop sudo tmux tree vim wget zsh \
    ninja-build libsparsehash-dev ffmpeg
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm --recursive --force /var/lib/apt/lists/*

# ====================================================
# user setup
# ====================================================

ARG UNAME=docker
ARG UID=1000
ARG GID=1000

RUN groupadd --gid ${GID} ${UNAME} && \
    useradd --uid ${UID} --gid ${GID} --create-home --groups sudo ${UNAME}
RUN echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${UNAME} \
    && chmod 440 /etc/sudoers.d/${UNAME}

USER ${UNAME}
ENV HOME="/home/${UNAME}"
WORKDIR ${HOME}

# ====================================================
# python env
# ====================================================

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/
RUN uv venv --python 3.10
ENV VIRTUAL_ENV="${HOME}/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0"
ENV FORCE_CUDA="1"
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    uv pip install -r requirements.txt
RUN uv pip install git+https://github.com/mit-han-lab/torchsparse.git@v2.0.0 --no-build-isolation
RUN uv pip install natten==0.17.1+torch210cu121 --find-links https://shi-labs.com/natten/wheels/ --no-build-isolation

WORKDIR ${HOME}/workspace