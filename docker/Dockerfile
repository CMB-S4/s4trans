
from menanteau/spt3g_ingest:ubuntu_0.3.3_4efaaf2f

user root
RUN apt-get update --fix-missing

# Git clone and version checkout and install of spt3g_ingest
ARG SPT3G_INGEST_VERSION
RUN git clone https://github.com/CMB-NCSA/s4trans -b $S4TRANS_INGEST_VERSION && \
    cd s4trans && \
    python setup.py install
