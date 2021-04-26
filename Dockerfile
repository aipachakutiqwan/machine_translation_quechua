FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
RUN mkdir -p /app
ADD requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Application layer

ADD configuration /app/configuration
ADD logs /app/logs
ADD src /app/src

# Creation of logs
RUN touch /app/logs/translator-logs.log
RUN chmod a+w /app/logs/translator-logs.log

WORKDIR /app
