import time
import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddlenlp
from functools import partial
from paddlenlp.data import Stack, Pad, Tuple
import json
import pandas as pd
import os

json_file_path = 'train.json'

if os.path.exists(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        # Load your own train, dev, and test datasets
        with open('train.json', 'r', encoding='utf-8') as f:
            train_ds = json.load(f)

        with open('dev.json', 'r', encoding='utf-8') as f:
            dev_ds = json.load(f)

        with open('test.json', 'r', encoding='utf-8') as f:
            test_ds = json.load(f)
else:
    print(f"'{json_file_path}' 未找到。")


# 输出训练集的前 3 条样本
for idx, example in enumerate(train_ds):
    if idx <= 3:
        print(example)

tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
# 将 1 条明文数据的 query、title 拼接起来，根据预训练模型的 tokenizer 将明文转换为 ID 数据
# 返回 input_ids 和 token_type_ids

# Convert example function
def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    query, title, label = example["query"], example["title"], example["label"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([label], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

### 对训练集的第 1 条数据进行转换
input_ids, token_type_ids, label = convert_example(train_ds[0], tokenizer)
print(input_ids)
print(token_type_ids)
print(label)
# 为了后续方便使用，我们使用python偏函数（partial）给 convert_example 赋予一些默认参数
from functools import partial
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
c = [5, 6, 7, 8]
result = Stack()([a, b, c])
print("Stacked Data: \n", result)
print()

a = [1, 2, 3, 4]
b = [5, 6, 7]
c = [8, 9]
result = Pad(pad_val=0)([a, b, c])
print("Padded Data: \n", result)
print()

data = [
        [[1, 2, 3, 4], [1]],
        [[5, 6, 7], [0]],
        [[8, 9], [1]],
       ]
batchify_fn = Tuple(Pad(pad_val=0), Stack())
ids, labels = batchify_fn(data)
print("ids: \n", ids)
print()
print("labels: \n", labels)
print()

trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=512)

# 我们的训练数据会返回 input_ids, token_type_ids, labels 3 个字段
# 因此针对这 3 个字段需要分别定义 3 个组 batch 操作
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]

# 定义分布式 Sampler: 自动对训练数据进行切分，支持多卡并行训练
batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)

# 基于 train_ds 定义 train_data_loader
# 因为我们使用了分布式的 DistributedBatchSampler, train_data_loader 会自动对训练数据进行切分
train_data_loader = paddle.io.DataLoader(
        dataset=train_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

# 针对验证集数据加载，我们使用单卡进行评估，所以采用 paddle.io.BatchSampler 即可
# 定义 dev_data_loader
batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=False)
dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

import paddle.nn as nn

# 我们基于 ERNIE-Gram 模型结构搭建 Point-wise 语义匹配网络
# 所以此处先定义 ERNIE-Gram 的 pretrained_model
pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
#pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained('ernie-1.0')


class PointwiseMatching(nn.Layer):
   
    # 此处的 pretained_model 在本例中会被 ERNIE-Gram 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # 语义匹配任务: 相似、不相似 2 分类任务
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        # 此处的 Input_ids 由两条文本的 token ids 拼接而成
        # token_type_ids 表示两段文本的类型编码
        # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)

        # 基于文本对的语义表示向量进行 2 分类任务
        logits = self.classifier(cls_embedding)
        probs = F.softmax(logits)

        return probs

# 定义 Point-wise 语义匹配网络
model = PointwiseMatching(pretrained_model)
from paddlenlp.transformers import LinearDecayWithWarmup

epochs = 3
num_training_steps = len(train_data_loader) * epochs

# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)

# 采用交叉熵 损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()
# 因为训练过程中同时要在验证集进行模型评估，因此我们先定义评估函数

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                    np.mean(losses), accu))
    model.train()
    metric.reset()
# 接下来，开始正式训练模型，训练时间较长，可注释掉这部分

# Training loop
global_step = 0
tic_train = time.time()

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        # Rest of the training loop code remains the same

        # Every 100 steps, evaluate on dev set
        if global_step % 100 == 0:
            evaluate(model, criterion, metric, dev_data_loader, "dev")
# 训练结束后，存储模型参数
save_dir = os.path.join("checkpoint", "model_%d" % global_step)
os.makedirs(save_dir)

save_param_path = os.path.join(save_dir, 'model_state.pdparams')
paddle.save(model.state_dict(), save_param_path)
tokenizer.save_pretrained(save_dir)

def predict(model, data_loader):
    
    batch_probs = []

    # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉
    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)
            
            # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
            batch_prob = model(
                input_ids=input_ids, token_type_ids=token_type_ids).numpy()

            batch_probs.append(batch_prob)
        batch_probs = np.concatenate(batch_probs, axis=0)

        return batch_probs
# 预测数据的转换函数
# predict 数据没有 label, 因此 convert_exmaple 的 is_test 参数设为 True
# Define your test dataset's transformation function
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=512, is_test=True)

# Define batchify function for prediction
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id)  # token_type_ids
): [data for data in fn(samples)]

# Define test data loader using BatchSampler
batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=32, shuffle=False)
test_data_loader = paddle.io.DataLoader(
    dataset=test_ds.map(trans_func),
    batch_sampler=batch_sampler,
    collate_fn=batchify_fn,
    return_list=True
)
# 生成预测数据 data_loader
predict_data_loader =paddle.io.DataLoader(
        dataset=test_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')

model = PointwiseMatching(pretrained_model)
# 刚才下载的模型解压之后存储路径为 ./ernie_gram_zh_pointwise_matching_model/model_state.pdparams
state_dict = paddle.load("./ernie_gram_zh_pointwise_matching_model/model_state.pdparams")

# 刚才下载的模型解压之后存储路径为 ./pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams
# state_dict = paddle.load("pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams")
model.set_dict(state_dict)
for idx, batch in enumerate(predict_data_loader):
    if idx < 1:
        print(batch)
# 执行预测函数
# Prediction loop
y_probs = predict(model, test_data_loader)

# Process prediction probabilities and save results

# 根据预测概率获取预测 label
y_preds = np.argmax(y_probs, axis=1)
