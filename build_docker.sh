

# Build container
export S4TRANS_VERSION=0.1.4
export S4USER=$USER
export IMAGE=s4trans
export TAG=${S4TRANS_VERSION}
docker build -f docker/Dockerfile \
       -t menanteau/$IMAGE:$TAG \
       --build-arg S4TRANS_VERSION \
       --build-arg S4USER \
       --no-cache \
       --rm=true .

echo 'Push commands:'
echo "   docker push menanteau/$IMAGE:${TAG}"

echo 'To create singularity image:'
echo "  ./docker2singularity menanteau/$IMAGE:${TAG}"
