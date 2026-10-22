from typing import TYPE_CHECKING
'''
paddleocr：解析图片中的文字，也可以进行表格识别
rapidocr_paddle 和 rapidocr_onnxruntime 两种导入方式
主要区别在于它们所使用的推理引擎和硬件支持
选择哪种方式最合适取决于你的硬件环境和性能需求。
当你有 GPU 且追求速度时：使用 rapidocr_paddle。PaddlePaddle 原生支持在 GPU 上推理 PaddleOCR 模型，速度更快。
当只有 CPU 且需要高效推理时：使用 rapidocr_onnxruntime（ONNX Runtime 是一个通用的推理引擎），。它在 CPU 上进行了优化，资源占用较低.
'''

# 进程内只加载一次。每个文件都 new 一次会反复把检测/识别权重从磁盘读进内存。
_ocr = None


def get_ocr(use_cuda: bool = True) -> 'RapidOCR': # type: ignore
    global _ocr
    if _ocr is not None:
        return _ocr
    try:
        from rapidocr_paddle import RapidOCR
        '''
        det_use_cuda=True：启用检测模型的GPU加速。cls_use_cuda=True：启用分类模型的GPU加速。rec_use_cuda=True：启用识别模型的GPU加速。
        '''
        _ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        #   cpu版本
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr
