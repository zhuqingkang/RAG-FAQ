import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from .edu_docloader import OCRDOCLoader
from .edu_pptloader import OCRPPTLoader
from .edu_imgloader import OCRIMGLoader
from .edu_pdfloader import OCRPDFLoader

__version__ = "0.1.0"
__author__ = "<zhuqingkang>"
__email__ = "<zhuqingkangactive@outlook.com>"
__license__ = "MIT"
__all__ = [
        'OCRDOCLoader',
       'OCRPPTLoader',
       'OCRIMGLoader',
       'OCRPDFLoader'
       ]
