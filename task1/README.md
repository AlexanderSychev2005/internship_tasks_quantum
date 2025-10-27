# ⛰️Text Mountain Classification

<img src="https://i.pinimg.com/1200x/d5/8d/b4/d58db4c5c90aa1972dc183dac9e5b4e9.jpg" height="500" alt="mountain" style="float:left; margin-right: 20px;">

### This project focuses on training (fine-tuning) BERT model and using it for NER (Named Entity Recognition).
### The task is to extract mountain names from the text. (e.g. "Everest", "Fuji"")

The NER model is trained on synthetic data.

<br style="clear:both;">

### Installation requirements:
1. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/Scripts/activate  
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### How to open the dataset creation notebook:
1. Make sure you have Jupyter Notebook installed.:
    ```bash
    pip install jupyter
    ```
2. Launch Jupyter Notebook:
    ```bash
    jupyter notebook dataset_creation.ipynb
    ```

### How to open the Demo Notebook:
1. Make sure you have Jupyter Notebook installed.:
    ```bash
    pip install jupyter
    ```
2. Launch Jupyter Notebook:
    ```bash
    jupyter notebook Demo.ipynb
    ```

### Project Structure:
```bash

task1/
├── dataset/
│   ├── dataset.json
├── dataset_creation.ipynb
├── Demo.ipynb
├── train_bert.py
├── infer_bert.py
├── requirements.txt
└── README.md
```
### Model Overview:
#### BERT for Named Entity Recognition (NER):
- **Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Task**: Named Entity Recognition (NER) to identify mountain names in text, using tokens like "B-MOUNTAIN", "I-MOUNTAIN", and "O". Synthetic data is generated using different templates.

#### Training Process:
```bash
python train_bert.py --dataset_path dataset/dataset.json --model_name bert-base-cased --output_dir ./models/ner_model_bert --num_of_epochs 5 --batch_size 16 --learning_rate 5e-5 --test_size 0.2
```
#### Inference Process:
```bash
python infer_bert.py --model_path ./models/ner_model_bert --test_sentence "Look! I love that mountain Fuji! And also the Everest is great."
```
Output example:
```
Entity: MOUNTAIN, Score: 0.9988, Word: Fuji
Entity: MOUNTAIN, Score: 0.9974, Word: Everest
Extracted mountains: ['Fuji', 'Everest']
```



