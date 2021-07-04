FROM python:3.9.2-slim-buster

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      make    \
      nano    \
      git &&  \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home
COPY requirements.txt /home
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt


COPY format_data.py /home/

ENTRYPOINT ["python3", "-u", "format_data.py"]
CMD ["--upload"]
