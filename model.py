from transformers import BertModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = r"E:\python_works\pytorch\huggingface\my_model_cache\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
model = BertModel.from_pretrained(model_name).to(device)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out