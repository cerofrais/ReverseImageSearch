FROM python:3.6-stretch


COPY . /app
WORKDIR /app/

RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN apt-get install libzbar-dev libzbar0 -y

RUN python3 -m pip install -r requirements.txt

EXPOSE 4050

CMD [ "sh","-c","python3 ignite_demo_app.py" ]

