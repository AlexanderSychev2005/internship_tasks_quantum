import json

from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import precision_score, recall_score, f1_score
from datasets import Dataset
import argparse

labels = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]

labels_map = {label: i for i, label in enumerate(labels)}  # Mapping labels to integers
id2label = {i: label for i, label in enumerate(labels)}
print(labels_map)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)


    true_labels = []
    true_preds = []
    for i in range(len(labels)):
        labels_list = labels[i]
        preds_list = preds[i]

        temp_true_labels = []
        temp_true_preds = []

        for j in range(len(labels_list)):
            if labels_list[j] != -100:  # Ignore predictions on padding tokens
                temp_true_labels.append(id2label[labels_list[j]])
                temp_true_preds.append(id2label[preds_list[j]])

        true_labels.append(temp_true_labels)
        true_preds.append(temp_true_preds)

    precision = precision_score(true_labels, true_preds)
    recall = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)

    return {"precision": precision, "recall": recall, "f1": f1}


def encode_data_bio(tokenizer, texts, entities):
    """
    Encodes the texts and the labels using the BIO structures.
    Tokenization with offsets, offset mapping is needed to align labels with tokens, returns start and end character
    positions of each token (start, end).
    truncation - truncate sequences to the model's maximum length
    padding - add padding tokens to make all sequences the same length

    :param tokenizer: Tokenizer to use for encoding the texts
    :param texts: List of input texts (sentences)
    :param entities: List of entities (start, end) for each text
    :return: Encoded texts and labels in BIO format, which is appropriate for token classification model
    """
    encodings = tokenizer(
        texts, truncation=True, return_offsets_mapping=True
    )
    all_labels = []

    for i, offsets in enumerate(encodings["offset_mapping"]):
        labels = []
        for start, end in offsets:
            if (
                start == 0 and end == 0
            ):  # [PAD] token, set label to -100, so it's ignored in loss computation
                labels.append(-100)
            else:
                labels.append(labels_map["O"])  # Not an entity, set label to O

        # Set labels B-MOUNTAIN for each entity, I-MOUNTAIN for the rest tokens in the entity
        for start_char, end_char in entities[i]:
            token_indices = [
                idx
                for idx, (s, e) in enumerate(offsets)
                if s >= start_char
                and e <= end_char  # Token is inside the entity or the entity itself
            ]
            if token_indices:
                labels[token_indices[0]] = labels_map["B-MOUNTAIN"]

                for idx in token_indices[1:]:
                    labels[idx] = labels_map["I-MOUNTAIN"]

        all_labels.append(labels)

    encodings["labels"] = all_labels
    encodings.pop("offset_mapping")
    return encodings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training the NER BERT model on synthetic data with mountains"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/dataset.json",
        help="Path to the dataset JSON file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-cased",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/ner_model_bert",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--num_of_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split",
    )

    args = parser.parse_args()

    # Loading dataset
    dataset = json.load(open(args.dataset_path, "r"))

    texts = [item["text"] for item in dataset]
    entities = [item["spans"] for item in dataset]

    # Tokenization and encoding the data
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    encodings = encode_data_bio(tokenizer, texts, entities)

    # Creating Dataset object and splitting into train and validation sets
    dataset = Dataset.from_dict(encodings)
    dataset_split = dataset.train_test_split(test_size=args.test_size)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    # Decoding example
    example = train_dataset[0]
    decoded_example = tokenizer.decode(example["input_ids"], skip_special_tokens=False)
    print(
        f"Decoded example: \n {decoded_example} \n with labels: \n {example['labels']} ."
    )

    model = BertForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(labels)
    )
    model.config.id2label = {i: label for i, label in enumerate(labels)}

    model.config.label2id = {label: i for i, label in enumerate(labels)}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_of_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        save_steps=200,
        eval_steps=200,
        logging_steps=50,
        learning_rate=args.learning_rate,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator= DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
