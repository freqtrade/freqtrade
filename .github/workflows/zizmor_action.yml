name: GitHub Actions Security Analysis with zizmor 🌈

on:
  push:
    branches:
      - develop
      - stable
  pull_request:
    branches:
      - develop
      - stable

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: false

permissions: {}

jobs:
  zizmor:
    name: Run zizmor 🌈
    runs-on: ubuntu-latest
    permissions:
      security-events: write # Required for upload-sarif (used by zizmor-action) to upload SARIF files.
      # contents: read         # Only needed for private repos. Needed to clone the repo.
      # actions: read          # Only needed for private repos. Needed for upload-sarif to read workflow run info.
    steps:
      - name: Checkout repository
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false

      - name: Run zizmor 🌈
        uses: zizmorcore/zizmor-action@6599ee8b7a49aef6a770f63d261d214911a7ce02 # v0.6.0
