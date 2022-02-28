CONTAINER_NAME=ml-pipeline-preprocessing
DOCKER_ID=amusanga
TAG_NAME=v1
docker build -t ${CONTAINER_NAME} .
docker tag ${CONTAINER_NAME} ${DOCKER_ID}/${CONTAINER_NAME}:${TAG_NAME}
# docker push ${DOCKER_ID}/${CONTAINER_NAME}:${TAG_NAME}