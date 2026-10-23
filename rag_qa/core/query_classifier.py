# -*- coding:utf-8-*-
# 导入标准库
import json
import os
import torch
import sys

# 设置 TensorBoard 日志目录，请替换为你想要的路径

os.environ["TENSORBOARD_LOGGING_DIR"] = "./bert_logs"

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f' current_dir--》{current_dir} ')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)

# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
# 导入日志
from base import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class QueryClassifier:
    def __init__(self, model_path = os.path.join(rag_qa_path, 'models','bert_query_classifier')):
        # 初始化模型路径
        self.model_path = model_path
        # 加载 BERT 分词器
        self.bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained("../models/bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # 初始化模型
        self.model = None
        # 确定设备（GPU 或 CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 记录设备信息
        # logger.info(f"使用设备: {self.device}")
        # 定义标签映射
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 加载模型
        self.load_model()

    # 目的:是加载原始的预训练模型(通义:中文文本分类)
    def load_model(self):
        # 检查模型路径是否存在
        if not os.path.exists(self.model_path): # 文件夹不存在
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.bert_path, num_labels=2)
            # 将模型移到指定设备
            self.model.to(self.device) # type:ignore
            # 记录加载成功的日志
            logger.info(f'加载预训练bert模型成功: {self.bert_path}')
            return

        #  初始化新模型
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=2)
        # 将模型移到指定设备
        self.model.to(self.device)# type:ignore
        # 记录加载成功的日志
        logger.info(f'加载bert微调模型成功: {self.model_path}')


    def train_model(self, data_file="model_generic_2000.json"):
        """
            1.数据的预处理：
                1.1 训练集，测试集，验证集 8，1，1
            2.模型训练
                学习率，批次，轮数，损失
            3. 模型保存
        """
        # 加载数据集
        # 1.判断文件是否存在
        with open(data_file, 'r', encoding='utf-8') as f:
             # 2.读取文件,并转成json
            lines : list[ dict ] = [json.loads(i) for i in f.readlines()]

        # 3.获取问题和标签
        # 3.1获取所有的问题(特征)
        x = [i['query'] for i in lines]
        # 3.2获取所有的标签
        y = [i['label'] for i in lines]

        # 4.数据划分
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        x_test,  x_valid,  y_test , y_valid  = train_test_split(x_test, y_test, test_size=0.5, shuffle=True)
        # print(f'train{len(x_train)/(len(x_test)+len(x_train)+len(x_valid))} , test{len(x_test)/(len(x_test)+len(x_train)+len(x_valid))} ,valid{len(x_valid)/(len(x_test)+len(x_train)+len(x_valid))}')
        # 5. 数据预处理:训练和测试,分词 + 获取标签类别[0,1]
        x_train ,y_train = self.preprocess_data(x_train,y_train)
        x_test ,y_test = self.preprocess_data(x_test,y_test)


        # 6.获取dataset对象: create_dataset
        dataset_train = self.create_dataset(x_train,y_train)
        dataset_test = self.create_dataset(x_test,y_test)


        # 7.设置训练参数
        training_args = TrainingArguments(
            output_dir="./bert_results",  # 模型（检查点）以及日志保存的路径等，
            num_train_epochs=3,  # 训练的轮次
            per_device_train_batch_size=8,  # 训练的批次
            per_device_eval_batch_size=8,  # 验证批次
            warmup_steps=20,  # 学习率预热的步数
            weight_decay=0.01,  # 权重衰减系数 学习率
            # logging_dir="./bert_logs",  # 日志保存路径:如果想生成这个文件夹，需要安装tensorboard
            report_to="tensorboard",
            logging_steps=10,  # 每隔多少步打印日志
            eval_strategy="epoch",  # 每轮都进行评估
            save_strategy="epoch",  # 每轮都进行检查点的模型保存
            load_best_model_at_end=True,  # 加载最优的模型
            save_total_limit=1,  # 只保存一个检查点，其他被覆盖
            metric_for_best_model="eval_loss",  # 评估最优模型的指标（验证集损失）,内部模型评估方式
            fp16=True
        )

        # 8. 初始化Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            compute_metrics=self.compute_metrics
        )

        # 9. 训练模型
        trainer.train()
        # # 10. 模型保存
        self.save_model()
        # 11. 对验证集进行训练好的模型验证
        self.evaluate_model(x_valid,y_valid)

    def preprocess_data(self, texts, labels):
        # pass
        """预处理数据为 BERT 输入格式"""
        # .tokenizer(xx,truncation=x,padding=xx,max_length=xx,return_tensors=xx)
        encodings =self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=128
        )
        labels = [self.label_map[i] for i in labels]

        return encodings, labels


    def save_model(self):

        """保存模型:save_pretrained"""
        self.model.save_pretrained(self.model_path) # type:ignore
        """保存分词器"""
        self.tokenizer.save_pretrained(self.model_path)
        # 日志打印
        logger.info(f'微调模型保存成功')


    def create_dataset(self, encodings, labels):

        # 自定义Dataset类
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                # 测试代码
                # for key, value in self.encodings.items():
                #     print(key, value)

                dicts = {key: value[idx] for key, value in self.encodings.items()}
                dicts['labels'] = torch.tensor(self.labels[idx])
                return dicts
        return Dataset(encodings,labels)

    def compute_metrics(self, eval_pred):
        """计算评估指标"""

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)  # 按行取最大值对应的下标id
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    """评估模型性能"""

    def evaluate_model(self, x_valid, y_valid):
        # 1.分词器
        encodings,labels = self.preprocess_data(x_valid, y_valid)
        # 2.获取数据集:create_dataset
        dataset_valid = self.create_dataset(encodings, labels)
        # 3. 加载训练模型:Trainer
        trainer = Trainer(model=self.model)
        # 4.模型预测
        logits = trainer.predict(dataset_valid)
        # 5. 获取标签:np.argmax
        pre_numpy = np.argmax(logits.predictions, axis = 1)
        # 6. 模型分类报告,混淆矩阵 : 比较真实标签和预测的标签
        logger.info(f'分类评估报告：{classification_report(labels, pre_numpy,target_names=["通用知识","专业咨询"])}')
        logger.info(f'混淆矩阵：{confusion_matrix(labels, pre_numpy)}')



    # 模型分类预测
    def predict_category(self, query):  # query:提示词
        # pass
        # 检查模型是否加载
        if self.model is None:
            # 模型未加载，记录错误
            logger.error('模型加载错误')
            # 默认返回通用知识
            return '通用知识'

        # 对查询进行编码
        encodings = self.tokenizer(
            query,
            truncation=True, padding=True,
            max_length=128,
            return_tensors='pt'
        )
        # 将编码移到指定设备
        encodings = {k:v.to(self.device) for k,v in encodings.items()}
        # 不计算梯度，进行预测
        with torch.no_grad():
            # 获取预测结果
            output = self.model(**encodings)
            pre = torch.argmax(output.logits, dim=1).item()
        # 根据预测结果返回类别
        return '专业知识' if pre ==1 else '通用知识'
