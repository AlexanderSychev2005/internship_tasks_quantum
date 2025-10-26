from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import torch
import argparse


def infer_bert(text, model_path):
    """
    Infer mountain entities from text using a fine-tuned BERT model.
    :param text: Input text containing mountain entities
    :param model_path: Path to the fine-tuned BERT model
    :return: List of extracted mountain entities
    """
    model = BertForTokenClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )
    results = []
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    ner_results = ner_pipeline(text)
    for result in ner_results:
        entity_group = result.pop("entity_group")
        score = result.pop("score")
        word = result.pop("word")
        word = word.strip(punctuation)

        results.append((entity_group, score, word))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer mountain entity from text using a fine-tuned BERT model"
    )
    parser.add_argument(
        "--test_sentence",
        type=str,
        default="Look! I love that mountain Fuji! And also the Everest is great.",
        help="Input text containing an mountain entity",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/ner_model_bert",
        help="Path to the fine-tuned BERT model",
    )
    args = parser.parse_args()

    results = infer_bert(args.test_sentence, args.model_path)
    for entity, score, word in results:
        print(f"Entity: {entity}, Score: {score:.4f}, Word: {word}")

    mountains = [word for entity, score, word in results]
    print(f"Extracted mountains: {mountains}")
