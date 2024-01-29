
TAGNAME=0.3.3
DOCKER_IMA=menanteau/s4trans:$TAGNAME
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
