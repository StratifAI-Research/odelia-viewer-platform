FROM docker.io/orthancteam/orthanc:25.12.0

RUN mkdir /python
COPY . /python/
RUN pip3 install --break-system-packages -r /python/requirements.txt
