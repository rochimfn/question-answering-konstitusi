# Indonesia Constitution Question Answering System (Telegram Bot, Streamlit Page, and HTTP API)

## *Quick Deploy*

This app uses `docker` and `docker-compose`.

0. Clone this repository with `git clone https://github.com/rochimfn/question-answering-konstitusi.git`.
1. Enter the app directory with `cd question-answering-konstitusi`,
2. Create `.env` file with `touch .env` (*nix) or `ni .env` (windows powershell).
3. Fill the `.env` with your Telegram bot token. Example: `BOT_TOKEN=xxx:xxxxxx`.
4. Build the image with `docker-compose build`.
5. Start the container with `docker-compose up -d`.
6. The HTTP API is served at port 8000 and the bot will be started soon (try /start command in your bot room).


## *Start Develop*

This app is developed in python version 3.8. Make sure you have it installed.

0. Clone this repository with `git clone https://github.com/rochimfn/question-answering-konstitusi.git`.
1. Enter the app directory with `cd question-answering-konstitusi`,
2. Create a virtual environment for python dependencies with `python -m venv venv` or `python -m virtualenv venv`.
3. Enter the virtual environment with `source venv/bin/activate` (*nix) or `venv/Scripts/activate` (windows).
4. Install all the dependencies with `pip install -r requirements.txt`.
5. Install the local share component with `pip install -e .`.
6. Set the required environment variables `BOT_TOKEN`, `ENABLE_PROOFING`, `NUM_RANK`, `QA_HOST`, and `QA_PORT`. Please refer to your operating system documentation on how to set environment variables.
7. Start the training with `python console/train.py` (Required). 
8. You can start the streamlit page with `streamlit run main.py` (Optional). 
9. You can start the HTTP API with `python web.py`(Optional). 
10. You can start the bot with `python bot.py` (Optional). Make sure to start the HTTP API first.
