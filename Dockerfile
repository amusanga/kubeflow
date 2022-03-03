FROM python:3.8

RUN apt-get update \
    && apt-get install git -y \
    && apt-get update && apt-get -y install cmake protobuf-compiler \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && apt install -y libgl1-mesa-glx \
    && apt-get install ffmpeg libsm6 libxext6  -y \
    && rm -rf /var/lib/apt/lists/*