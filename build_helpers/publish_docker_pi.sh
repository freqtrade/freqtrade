#!/bin/sh

# The below assumes a correctly setup docker buildx environment

# Replace / with _ to create a valid tag
TAG=$(echo "${BRANCH_NAME}" | sed -e "s/\//_/g")
PI_PLATFORM="linux/arm/v7"
echo "Running for ${TAG}"
CACHE_TAG=freqtradeorg/freqtrade_cache:${TAG}_cache

# Add commit and commit_message to docker container
echo "${GITHUB_SHA}" > freqtrade_commit

if [ "${GITHUB_EVENT_NAME}" = "schedule" ]; then
    echo "event ${GITHUB_EVENT_NAME}: full rebuild - skipping cache"
    docker buildx build \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG} --push .
else
    echo "event ${GITHUB_EVENT_NAME}: building with cache"
    # Pull last build to avoid rebuilding the whole image
    # docker pull --platform ${PI_PLATFORM} ${IMAGE_NAME}:${TAG}
    docker buildx build \
        --cache-from=type=registry,ref=${CACHE_TAG} \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG} --push .
fi

if [ $? -ne 0 ]; then
    echo "failed building image"
    return 1
fi
