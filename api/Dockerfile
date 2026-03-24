FROM continuumio/miniconda3

RUN apt-get update -y 

RUN apt-get install git -y

WORKDIR /home/app

RUN git clone https://github.com/YoannRobert/DDay_FullStack.git

WORKDIR /home/app/DDay_FullStack

RUN git fetch

RUN git checkout develop

RUN conda env create -f environment.yml

RUN echo "conda activate dday" >> ~/.bashrc

WORKDIR /home/app/DDay_FullStack/api 

ENTRYPOINT ["conda", "run", "-n", "dday", "python", "main.py"]