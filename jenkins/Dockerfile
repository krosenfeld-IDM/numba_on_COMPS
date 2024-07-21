FROM rockylinux:9

COPY requirements.txt /requirements.txt

RUN dnf upgrade -y
RUN dnf install -y \
    python \
    python-pip \
    python-devel \
    vim \
    git \
    sudo

RUN pip install pip --upgrade && pip install -r requirements.txt
