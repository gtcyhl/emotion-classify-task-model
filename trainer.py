import torch
from dataset import EmotionDataset
from torch.utils.data import DataLoader
from model import Model
from transformers import BertTokenizer
from torch.optim import Adam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 100
token = BertTokenizer.from_pretrained("bert-base-chinese")

# 自定义函数，对数据进行编码处理
def collate_fn(data):
    # 文本
    sentes = [i[0] for i in data]
    # 分类
    label = [i[1] for i in data]
    # 编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sentes,
        truncation=True,
        padding="max_length",
        max_length=350,
        return_tensors="pt",
        return_length=True,
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, labels

# 获取训练集
train_dataset = EmotionDataset("train")
# 使用DataLoader转换
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32, # 核心参数，批次理论越大效果越好
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

if __name__ == '__main__':
    model = Model().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=5e-4) # 优化器
    loss_func = torch.nn.CrossEntropyLoss() # 损失函数

    model.train()
    break_flag = 0
    for epoch in range(EPOCH):
        for i,(input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 将数据放到DEVICE上
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            # 前向传播
            out = model(input_ids, attention_mask, token_type_ids)
            # 损失计算
            loss = loss_func(out, labels)
            # 反向更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%5 ==0:
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item()/len(labels)
                print(epoch, i, loss.item(), acc)
                if acc > 0.9:
                    # 保存模型参数
                    torch.save(model.state_dict(), "./bert.pt")
                    break_flag = 1
                    break
        if break_flag == 1:
            break