
TAGNAME=ubuntu_0.3.3_4efaaf2f
DOCKER_IMA=menanteau/spt3g_ingest:$TAGNAME
NAME=cmbs4
hostname="`hostname -s`-$NAME"

docker run -ti \
       -h $hostname\
       -v $HOME/CMBS4dev/cmb-home:/home/felipe \
       --name $NAME \
       $DOCKER_IMA bash

# To re-enter from
# NAME=cmbs4; docker exec -ti $NAME bash

# To clean up
# docker rm $(docker ps -a -f status=exited -q)
