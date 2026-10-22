import os, sys


# 把项目根目录 integrated_qa_system 加入 sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base import Config, logger

logger.info(Config().LOG_FILE)