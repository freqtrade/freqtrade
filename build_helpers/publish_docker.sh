#!/bin/sh
# - export TAG=`if [ "$TRAVIS_BRANCH" == "develop" ]; then echo "latest"; else echo $TRAVIS_BRANCH ; fi`
# Replace / with _ to create a valid tag
TAG=$(echo "${TRAVIS_BRANCH}" | sed -e "s/\//_/")


# Add commit and commit_message to docker container
echo "${TRAVIS_COMMIT} ${TRAVIS_COMMIT_MESSAGE}" > freqtrade_commit

if [ "${TRAVIS_EVENT_TYPE}" = "cron" ]; then
    echo "event ${TRAVIS_EVENT_TYPE}: full rebuild - skipping cache"
    docker build -t freqtrade:${TAG} .
else
    echo "event ${TRAVIS_EVENT_TYPE}: building with cache"
    # Pull last build to avoid rebuilding the whole image
    docker pull ${IMAGE_NAME}:${TAG}
    docker build --cache-from ${IMAGE_NAME}:${TAG} -t freqtrade:${TAG} .
fi

if [ $? -ne 0 ]; then
    echo "failed building image"
    return 1
fi

# Run backtest
docker run --rm -it -v $(pwd)/config.json.example:/freqtrade/config.json:ro -v $(pwd)/tests:/tests freqtrade:${TAG} backtesting --datadir /tests/testdata --strategy DefaultStrategy

if [ $? -ne 0 ]; then
    echo "failed running backtest"
    return 1
fi

# Tag image for upload
docker tag freqtrade:$TAG ${IMAGE_NAME}:$TAG
if [ $? -ne 0 ]; then
    echo "failed tagging image"
    return 1
fi

# Tag as latest for develop builds
if [ "${TRAVIS_BRANCH}" = "develop" ]; then
    docker tag freqtrade:$TAG ${IMAGE_NAME}:latest
fi

# Login
echo "$DOCKER_PASS" | docker login -u $DOCKER_USER --password-stdin

if [ $? -ne 0 ]; then
    echo "failed login"
    return 1
fi

# Show all available images
docker images

docker push ${IMAGE_NAME}
if [ $? -ne 0 ]; then
    echo "failed pushing repo"
    return 1
fi
