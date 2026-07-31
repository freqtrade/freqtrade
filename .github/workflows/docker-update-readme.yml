name: Update Docker Hub Description
on:
  push:
    branches:
    - stable
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}
  cancel-in-progress: true

# disable permissions for all of the available permissions
permissions: {}

jobs:
  dockerHubDescription:
    name: "Update Docker Hub Description"
    runs-on: ubuntu-latest
    environment:
      name: docker
    steps:
    - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      with:
        persist-credentials: false

    - name: Docker Hub Description
      uses: peter-evans/dockerhub-description@1b9a80c056b620d92cedb9d9b5a223409c68ddfa # v5.0.0
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
        repository: freqtradeorg/freqtrade
