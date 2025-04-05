import torch
from model import Model
from transformers import BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names = ["负向评价","正向评价"]
model = Model().to(DEVICE)
model_name = r"E:\python_works\pytorch\huggingface\my_model_cache\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
token = BertTokenizer.from_pretrained(model_name)

def collate_fn(data):
    sentes = []
    sentes.append(data)
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentes,
                            truncation=True,
                            padding="max_length",
                            max_length=500,
                            return_tensors="pt",
                            return_length=True)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    return input_ids,attention_mask,token_type_ids

def test():
    model.load_state_dict(torch.load("./bert.pt"))
    model.eval() # 评估模式
    while True:
        data = input("请输入测试数据(输入'q'退出)：")
        if data == "q":
            print("测试结束")
            break
        input_ids, attention_mask, token_type_ids= collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(
            DEVICE), token_type_ids.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
            out = out.argmax(dim=1)
            print("模型判定：",names[out],"\n")
if __name__ == '__main__':
    test()
