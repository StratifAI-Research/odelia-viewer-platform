FROM docker.io/orthancteam/orthanc:25.12.0

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

RUN mkdir /python
COPY . /python/
RUN pip3 install --break-system-packages -r /python/requirements.txt
