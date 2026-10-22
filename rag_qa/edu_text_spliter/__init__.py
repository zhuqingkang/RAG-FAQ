   # 如果相对导入失败，尝试绝对导入
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from edu_model_text_spliter import *
from edu_chinese_recursive_text_splitter import *