FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py"]

