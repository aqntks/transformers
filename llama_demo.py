from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    BertTokenizer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np


def get_dataset(tokenizer):
    # 가상의 학습 데이터
    train_texts = ["I love Hugging Face!", "I hate rainy days."]
    train_labels = [1, 0]

    # 입력 데이터를 토큰화하고 인코딩합니다.
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                   torch.tensor(train_encodings['attention_mask']),
                                                   torch.tensor(train_labels))

    return train_dataset


def peft(model):
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def main():
    model_checkpoint = "decapoda-research/llama-7b-hf"
    lr = 1e-3
    batch_size = 1
    num_epochs = 10
    weight_decay = 0.01

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = get_dataset(tokenizer)

    model = peft(model)

    training_args = TrainingArguments(
        output_dir="llama_sequence_classification",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()
