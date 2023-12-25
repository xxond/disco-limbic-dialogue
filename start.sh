if [ $# -eq 0 ]
  then
    echo "No arguments supplied, starting cmd client"
    python3 client_cmd.py
  else
    export TG_SECRET=$1
    echo "Starting telegram bot client"
    python3 client_tg_bot.py
fi



