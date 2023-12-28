# Project Overview

This project showcases an AI model meticulously trained on selected dialogues from "Disco Elysium", a critically acclaimed video game. This model replicates the game's unique narrative style, renowned for its blunt and forthright dialogue.

# Technical Specifications
For a detailed understanding of the model's framework and capabilities, please refer to the  [Model card](https://huggingface.co/xxond/disco-limbic-dialogue).

# Data

The raw data, sourced directly from the game, underwent a transformation into a graph structure, capturing the intricate connections between dialogue elements. The following visuals illustrate the data's interconnected nature:
![Connected part of graph](/assets/plot2d.png)
![Multiple parts of graph](/assets/plot3d.png)

The dialogue sampling adhered to specific criteria:

- Uniform frequency across all dialogue lines
- Alternating dialogue roles, primarily using `Harrier Du Bois` as the input and other characters as the output
- Exclusive focus on the protagonist's inner voices, such as `Empathy`, `Inland Empire`, and `Electrochemistry`

# Local install
The project has been tested with `Python 3.10`. To install the required dependencies:
```
pip install -r requirements.txt
```
For the command-line client, execute:
```
python client_cmd.py
```
For utilizing the Telegram bot client, firstly register a secret key with [Bot Father](https://t.me/BotFather). Then, set up the environment and run the client:
```
export TG_SECRET=<your secret key>
python client_tg_bot.py
```

# Docker Deployment
For the command-line client via Docker:
```
volume=$PWD/docker_model

docker run --gpus all -ti -v $volume:/root/.cache/ ghcr.io/xxond/disco-llm:v1
```
For deploying the Telegram bot client using Docker:
```
volume=$PWD/docker_model
secret=<your secret key>

docker run --gpus all -ti -v $volume:/root/.cache/ ghcr.io/xxond/disco-llm:v1 $secret
```

# Future Enhancements
The model performs admirably when dialogues remain within the character's persona or the game's logic. However, it shows instability with out-of-context inputs. A potential improvement could be incorporating responses such as "I don't understand you" for such inputs, enhancing the model's adaptability.

# Contact and Legal Disclaimer

For further details on the project's methodology, or to address any legal queries, please feel free to contact me. While the legal aspects of utilizing game dialogue for AI training are still being explored, your input and inquiries are welcomed to ensure compliance and ethical development.
