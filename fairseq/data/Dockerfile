FROM docker.io/python:3.10

#RUN mkdir /root/.DANE /mnt/dane-fs /src /model
WORKDIR /src

RUN pip install --upgrade pip
RUN pip install torch=2.1.2
RUN pip install numpy=1.26.2
RUN pip install opencv-python
RUN pip install pillow

COPY /home/kdeboer/sseft/. /src
#RUN preprocess
RUN python3 validate.py --data /test_data --path checkpoints/checkpoint_best.pt --task emotion_prediction --valid-subset valid --batch-size 1 --eval-matric --softmax-target-binary-meld
#ENTRYPOINT ["./docker-entrypoint.sh"] Hier word dane worker aangeroepen