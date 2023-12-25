FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

COPY ./requirements.txt /home/app/requirements.txt
WORKDIR /home/app

RUN pip install -r requirements.txt

COPY ./utils /home/app/utils

COPY ./client_tg_bot.py /home/app/client_tg_bot.py
COPY ./client_cmd.py /home/app/client_cmd.py
COPY ./start.sh /home/app/start.sh
COPY ./data/actors.txt /home/app/data/actors.txt

RUN chmod +x start.sh
ENTRYPOINT ["sh", "./start.sh"]