#!/bin/sh

# Replace / with _ to create a valid tag
TAG=$(echo "${BRANCH_NAME}" | sed -e "s/\//_/g")
TAG_PLOT=${TAG}_plot
echo "Running for ${TAG}"

# Add commit and commit_message to docker container
echo "${GITHUB_SHA}" > freqtrade_commit

if [ "${GITHUB_EVENT_NAME}" = "schedule" ]; then
    echo "event ${GITHUB_EVENT_NAME}: full rebuild - skipping cache"
    docker build -t freqtrade:${TAG} .
else
    echo "event ${GITHUB_EVENT_NAME}: building with cache"
    # Pull last build to avoid rebuilding the whole image
    docker pull ${IMAGE_NAME}:${TAG}
    docker build --cache-from ${IMAGE_NAME}:${TAG} -t freqtrade:${TAG} .
fi
# Tag image for upload and next build step
docker tag freqtrade:$TAG ${IMAGE_NAME}:$TAG

docker build --cache-from freqtrade:${TAG} --build-arg sourceimage=${TAG} -t freqtrade:${TAG_PLOT} -f docker/Dockerfile.plot .

docker tag freqtrade:$TAG_PLOT ${IMAGE_NAME}:$TAG_PLOT

if [ $? -ne 0 ]; then
    echo "failed building image"
    return 1
fi

# Run backtest
docker run --rm -v $(pwd)/config_bittrex.json.example:/freqtrade/config.json:ro -v $(pwd)/tests:/tests freqtrade:${TAG} backtesting --datadir /tests/testdata --strategy-path /tests/strategy/strats/ --strategy DefaultStrategy

if [ $? -ne 0 ]; then
    echo "failed running backtest"
    return 1
fi

if [ $? -ne 0 ]; then
    echo "failed tagging image"
    return 1
fi

# Tag as latest for develop builds
if [ "${TAG}" = "develop" ]; then
    docker tag freqtrade:$TAG ${IMAGE_NAME}:latest
fi

# Show all available images
docker images

docker push ${IMAGE_NAME}
if [ $? -ne 0 ]; then
    echo "failed pushing repo"
    return 1
fi
