import torch 


class BERTDatasetTraining:
    def __init__(self, qtitle, qbody, answer, targets, tokenizer, max_len):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, item):
        question_title = str(self.qtitle[item])
        question_body = str(self.qbody[item])
        answer = str(self.answer[item])

        # [CLS] [question_title] [question_body] [SEP] [question_answer] [SEP]
        inputs = self.tokenizer.encode_plus(
            question_title + " " + question_body, 
            answer,
            add_special_tokens=True, 
            max_len = self.max_len
        )

        ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        mask = inputs['attention_mask']

        padding_len = self.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)

        return{
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item, :], dtype=torch.float)
        }
