FROM docker.io/orthancteam/orthanc:25.12.0

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install pydicom --break-system-packages
RUN pip3 install dicomweb-client --break-system-packages
RUN pip3 install matplotlib numpy opencv-python Pillow requests --break-system-packages

RUN mkdir /python
COPY . /python/
