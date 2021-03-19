import numpy as np
import pandas as pd 

from sklearn import model_selection
from sklearn import metrics
from scipy import stats

import torch 
import transformers

import dataset
import Engine
import model_file  

from transformers import AdamW, get_linear_schedule_with_warmup 
import warnings
warnings.filterwarnings('ignore')

def run():

    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4
    TEST_BATCH_SIZE = 2
    EPOCHS = 20

    dfx = pd.read_csv('/content/drive/My Drive/Colab Notebooks/train.csv')     
    df_train, df_valid = model_selection.train_test_split(dfx, random_state=42, test_size=0.1)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    sample = pd.read_csv("/content/drive/My Drive/Colab Notebooks/sample_submission.csv")
    target_cols = list(sample.drop("qa_id", axis=1).columns)
    train_targets = df_train[target_cols].values
    valid_targets = df_valid[target_cols].values

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')


    train_dataset = dataset.BERTDatasetTraining(
        qtitle=df_train.question_title.values,
        qbody = df_train.question_body.values,
        answer = df_train.answer.values,
        targets = train_targets,
        tokenizer = tokenizer,
        max_len = MAX_LEN
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True 
    )

    valid_dataset = dataset.BERTDatasetTraining(
        qtitle=df_valid.question_title.values,
        qbody = df_valid.question_body.values,
        answer = df_valid.answer.values,
        targets = valid_targets,
        tokenizer = tokenizer,
        max_len = MAX_LEN
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = TEST_BATCH_SIZE 
    )


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_file.BERTBaseUncased('bert-based-uncased')
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )


    for epoch in range(EPOCHS):
        Engine.train_loop_fn(train_data_loader, model, optimizer, device, scheduler)
        output, target = Engine.eval_loop_fn(valid_data_loader, model, device)

        spear = []
        for j in range(target.shape[1]):
            p1 = list(target[:, j])
            p2 = list(output[:, j])
            coef, _ = np.nan_to_num(stats.spearmanr(p1, p2))
            spear.append(coef)
        spear = np.mean(spear)
        print(f"epoch = {epoch}, spearman = {spear}")
        torch.save(model.state_dict(), "model.bin")


if __name__ == "__main__":
    run()
