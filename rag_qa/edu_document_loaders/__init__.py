# rag_qa/edu_document_loaders/__init__.py
# 各 loader 之间用相对导入（from .edu_ocr import get_ocr），不再需要改 sys.path。
# 对外：from rag_qa.edu_document_loaders import OCRPDFLoader
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
