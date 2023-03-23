
from menanteau/spt3g_ingest:ubuntu_0.3.6_6149ab91

user root
RUN apt-get update --fix-missing

ARG PRODUCT_DIR=/opt/s4trans

# Git clone and version checkout and install of spt3g_ingest
ARG S4TRANS_VERSION
RUN git clone https://github.com/CMB-S4/s4trans -b $S4TRANS_VERSION && \
    cd s4trans && \
    python setup.py install --prefix=$PRODUCT_DIR --install-lib=$PRODUCT_DIR/python && \
    cd .. && \
    rm -rf s4trans

# Need this one to start automatically
RUN echo "echo Adding s4trans" >> /opt/spt/setSPTEnv.sh
RUN echo "source /opt/s4trans/setpath.sh /opt/s4trans" >> /opt/spt/setSPTEnv.sh

COPY --chown=${S4USER}:${S4USER} docker/startup.sh /opt/startup.sh
RUN chmod a+x /opt/startup.sh

# Add $SPTUSER as user and create groups wheel and spt
ARG S4USER
USER $S4USER
WORKDIR /home/$S4USER

# Need this one for bash
RUN echo "echo Adding s4trans" >> $HOME/.bashrc
RUN echo "source /opt/s4trans/setpath.sh /opt/s4trans" >> $HOME/.bashrc

ENTRYPOINT ["bash","/opt/startup.sh"]
