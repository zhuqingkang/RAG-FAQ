from langchain_text_splitters import CharacterTextSplitter
import importlib
import os
import re
import sys
import types
from typing import List
import transformers

# 直接运行本文件时，sys.path[0] 是当前脚本目录（edu_text_spliter），
# 找不到 integrated_qa_system 下的 base 包，因此需要把项目根目录加入 sys.path。
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from base import Config

# transformers>=4.48 移除了 transformers.onnx（ONNX 导出已迁移到 optimum），
# 而 modelscope 的 nlp/bert/configuration.py 仍直接 `from transformers.onnx import OnnxConfig`，
# 这里补一个兼容模块，避免 pipeline 加载时 ImportError。推理流程不会用到 OnnxConfig。
if 'transformers.onnx' not in sys.modules:
    try:
        importlib.import_module('transformers.onnx')
    except ModuleNotFoundError:
        _onnx_compat = types.ModuleType('transformers.onnx')

        class OnnxConfig:  # noqa: D401 仅作为 modelscope ONNX 导出类的基类占位
            pass

        setattr(_onnx_compat, 'OnnxConfig', OnnxConfig)
        sys.modules['transformers.onnx'] = _onnx_compat
        setattr(transformers, 'onnx', _onnx_compat)

from modelscope.pipelines import pipeline

# transformers 5 的 PretrainedConfig 不再为子类自动补这些通用属性，
# 而 modelscope 的 BertConfig 没有显式设置它们，这里补默认值。
from modelscope.models.nlp.bert.configuration import BertConfig as _MSBertConfig

if not hasattr(_MSBertConfig, '_edurag_compat_patched'):
    _orig_bert_init = _MSBertConfig.__init__

    _DEFAULT_ATTRS = {
        'is_decoder': False,
        'add_cross_attention': False,
        'chunk_size_feed_forward': 0,
        'output_attentions': False,
        'output_hidden_states': False,
    }

    def _patched_bert_init(self, *args, **kwargs):
        _orig_bert_init(self, *args, **kwargs)
        for key, default in _DEFAULT_ATTRS.items():
            if key not in self.__dict__:
                setattr(self, key, kwargs.get(key, default))
        if 'use_return_dict' not in self.__dict__ and not hasattr(type(self), 'use_return_dict'):
            self.use_return_dict = kwargs.get('return_dict', True)

    setattr(_MSBertConfig, '__init__', _patched_bert_init)
    setattr(_MSBertConfig, '_edurag_compat_patched', True)

# transformers 5 的 from_pretrained 需要访问 all_tied_weights_keys，
# modelscope 的 BertPreTrainedModel 未定义该属性，补一个空映射。
from modelscope.models.nlp.bert.backbone import BertPreTrainedModel as _MSBertPreTrainedModel

if not hasattr(_MSBertPreTrainedModel, 'all_tied_weights_keys'):
    setattr(_MSBertPreTrainedModel, 'all_tied_weights_keys', {})


# transformers 5 移除了 PreTrainedModel 上的 get_extended_attention_mask / invert_attention_mask /
# get_head_mask，modelscope 的 BertModel 仍在调用，这里按 transformers 4 的语义补回。
def _get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    import torch

    if dtype is None:
        # attention_mask 通常是整型，掩码计算需要用模型参数的浮点 dtype
        dtype = getattr(self, 'dtype', None) or torch.float32
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if getattr(self.config, 'is_decoder', False):
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            causal_mask = causal_mask.to(attention_mask.dtype)
            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat([
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device,
                               dtype=causal_mask.dtype),
                    causal_mask,
                ], dim=-1)
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f'Wrong shape for input_ids (shape {input_shape}) or attention_mask '
                         f'(shape {attention_mask.shape})')
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    return (1.0 - extended_attention_mask) * torch.finfo(dtype).min


def _invert_attention_mask(self, encoder_attention_mask):
    import torch

    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    else:
        raise ValueError(f'Wrong shape for encoder_attention_mask (shape {encoder_attention_mask.shape})')
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
    return (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min


def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    import torch

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        if head_mask.dim() != 5:
            raise ValueError('head_mask should be 1d, 2d or 5d')
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


for _method_name, _method in (
    ('get_extended_attention_mask', _get_extended_attention_mask),
    ('invert_attention_mask', _invert_attention_mask),
    ('get_head_mask', _get_head_mask),
):
    if not hasattr(_MSBertPreTrainedModel, _method_name):
        setattr(_MSBertPreTrainedModel, _method_name, _method)

# transformers 5 把 _keys_to_ignore_on_load_* 当集合使用（`list | set`），
# modelscope 里是 list，统一转成 set。
from modelscope.models.nlp.bert import backbone as _ms_bert_backbone
from modelscope.models.nlp.bert import document_segmentation as _ms_docseg

for _mod in (_ms_bert_backbone, _ms_docseg):
    for _obj in vars(_mod).values():
        if isinstance(_obj, type) and issubclass(_obj, _MSBertPreTrainedModel):
            for _attr in ('_keys_to_ignore_on_load_unexpected', '_keys_to_ignore_on_load_missing'):
                _val = getattr(_obj, _attr, None)
                if isinstance(_val, list):
                    setattr(_obj, _attr, set(_val))

# pipeline 会把 trust_remote_code 等通用参数透传给 preprocessor，
# 而 DocumentSegmentationTransformersPreprocessor 的签名不接受它们。
from modelscope.preprocessors.nlp.document_segmentation_preprocessor import (
    DocumentSegmentationTransformersPreprocessor as _MSDocSegPreprocessor,
)

if not hasattr(_MSDocSegPreprocessor, '_edurag_compat_patched'):
    _orig_pre_init = _MSDocSegPreprocessor.__init__

    def _patched_pre_init(self, *args, **kwargs):
        kwargs.pop('trust_remote_code', None)
        _orig_pre_init(self, *args, **kwargs)

    setattr(_MSDocSegPreprocessor, '__init__', _patched_pre_init)
    setattr(_MSDocSegPreprocessor, '_edurag_compat_patched', True)


class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 依赖版本需互相兼容，推荐：modelscope==1.33.0 transformers>=4.51,<5 datasets>=2.19,<3 huggingface-hub>=0.26,<1
        # pip install "modelscope==1.33.0" "transformers>=4.51,<5" "datasets>=2.19,<3" "huggingface-hub>=0.26,<1"
        # 考虑到使用了三个模型，可能对于低配置gpu不太友好，因此这里将模型load进cpu计算，有需要的话可以替换device为自己的显卡id
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub("\n\n", "", text)
        p = pipeline(
            task="document-segmentation",
            # modelscope 只接受 str 形式的模型路径，Config().Pre_bert 是 pathlib.Path
            model=str(Config().Pre_bert), # type: ignore
            device="cpu")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list
if __name__ == '__main__':
    model_split = AliTextSplitter(pdf=True)
    result = model_split.split_text(text='移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')
    print(result)