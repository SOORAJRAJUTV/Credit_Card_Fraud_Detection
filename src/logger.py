import logging
import os
from datetime import datetime

LOG_FILE  = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" ## log filename

logs_dir = os.path.join(os.getcwd(), "logs")  # Directory only
os.makedirs(logs_dir, exist_ok=True)  # Create the logs directory if not exists

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)  # Correct log file path


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__ == "__main__":
#      logging.info("Logging has started")

