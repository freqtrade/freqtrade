#!/bin/sh

# The below assumes a correctly setup docker buildx environment

# Replace / with _ to create a valid tag
TAG=$(echo "${BRANCH_NAME}" | sed -e "s/\//_/g")
TAG_PLOT=${TAG}_plot
TAG_PI="${TAG}_pi"

PI_PLATFORM="linux/arm/v7"
echo "Running for ${TAG}"
CACHE_TAG=freqtradeorg/freqtrade_cache:${TAG}_cache

# Add commit and commit_message to docker container
echo "${GITHUB_SHA}" > freqtrade_commit

if [ "${GITHUB_EVENT_NAME}" = "schedule" ]; then
    echo "event ${GITHUB_EVENT_NAME}: full rebuild - skipping cache"
    # Build regular image
    docker build -t freqtrade:${TAG} .
    # Build PI image
    docker buildx build \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f docker/Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG_PI} --push .
else
    echo "event ${GITHUB_EVENT_NAME}: building with cache"
    # Build regular image
    docker pull ${IMAGE_NAME}:${TAG}
    docker build --cache-from ${IMAGE_NAME}:${TAG} -t freqtrade:${TAG} .

    # Pull last build to avoid rebuilding the whole image
    # docker pull --platform ${PI_PLATFORM} ${IMAGE_NAME}:${TAG}
    docker buildx build \
        --cache-from=type=registry,ref=${CACHE_TAG} \
        --cache-to=type=registry,ref=${CACHE_TAG} \
        -f docker/Dockerfile.armhf \
        --platform ${PI_PLATFORM} \
        -t ${IMAGE_NAME}:${TAG_PI} --push .
fi

if [ $? -ne 0 ]; then
    echo "failed building multiarch images"
    return 1
fi
# Tag image for upload and next build step
docker tag freqtrade:$TAG ${IMAGE_NAME}:$TAG

docker build --cache-from freqtrade:${TAG} --build-arg sourceimage=${TAG} -t freqtrade:${TAG_PLOT} -f docker/Dockerfile.plot .

docker tag freqtrade:$TAG_PLOT ${IMAGE_NAME}:$TAG_PLOT

# Run backtest
docker run --rm -v $(pwd)/config_examples/config_bittrex.example.json:/freqtrade/config.json:ro -v $(pwd)/tests:/tests freqtrade:${TAG} backtesting --datadir /tests/testdata --strategy-path /tests/strategy/strats/ --strategy DefaultStrategy

if [ $? -ne 0 ]; then
    echo "failed running backtest"
    return 1
fi

docker images

docker push ${IMAGE_NAME}
docker push ${IMAGE_NAME}:$TAG_PLOT
docker push ${IMAGE_NAME}:$TAG

# Create multiarch image
# Make sure that all images contained here are pushed to github first.
# Otherwise installation might fail.

docker manifest create freqtradeorg/freqtrade:${TAG} ${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:${TAG_PI}
docker manifest push freqtradeorg/freqtrade:${TAG}

# Tag as latest for develop builds
if [ "${TAG}" = "develop" ]; then
    docker manifest create freqtradeorg/freqtrade:latest ${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:${TAG_PI}
    docker manifest push freqtradeorg/freqtrade:latest
fi


docker images

if [ $? -ne 0 ]; then
    echo "failed building image"
    return 1
fi
