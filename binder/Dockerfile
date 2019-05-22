FROM python:3.6

RUN apt-get update && apt-get install -y -q \ 
    mpich git apt-utils \ 
    libmpich-dev 

RUN pip install --no-cache notebook

### create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
##############

ADD ./requirements.txt /app/requirements.txt

RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir jupyter && \
    /venv/bin/pip install --no-cache-dir -r /app/requirements.txt

ADD ./devito /app/devito
ADD ./tests /app/tests
ADD ./examples /app/examples

ADD docker/run-jupyter.sh /jupyter
ADD docker/run-tests.sh /tests
ADD docker/run-print-defaults.sh /print-defaults
ADD docker/entrypoint.sh /docker-entrypoint.sh

RUN chmod +x \
    /print-defaults \
    /jupyter \
    /tests \
    /docker-entrypoint.sh

WORKDIR /app

ENV DEVITO_ARCH="gcc"
ENV DEVITO_OPENMP="0"

EXPOSE 8888
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/jupyter"]
