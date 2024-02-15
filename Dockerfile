FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN pip install pandas
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install opencv-python==4.8.1.78
RUN pip install tensorboard
RUN pip install torchsummary

RUN apt update
RUN apt install libgl1-mesa-glx -y
RUN apt install libglib2.0-0 -y
RUN apt install vim -y

COPY train.py /workspace/train.py
COPY dataset.py /workspace/dataset.py
COPY model.py /workspace/model.py