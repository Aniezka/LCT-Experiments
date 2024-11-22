# Advancing Multilingual Fact Checking with LLMs and Data Augmentation Techniques

This project will build upon the X-FACT multilingual dataset and baseline models presented in (Gupta et al., 2021). X-FACT is a large publicly available multilingual dataset for factual verification of real-world claims, spanning 25 typologically diverse languages across 11 language families. The dataset provides a benchmark for evaluating the cross-lingual transfer and generalization abilities of multilingual fact-checking systems.
While the original X-FACT paper implemented baseline models using multilingual BERT (Devlin et al., 2018), this project aims to extend that work by:
1) Experimenting with newer, more powerful multilingual language models
2) Investigating cross-lingual transfer learning between related languages
3) Augmenting the training data with synthetic examples through translation and paraphrasing

## 1. Data
The data used in this project is from the X-FACT dataset: https://huggingface.co/datasets/utahnlp/x-fact.

Experiments were conducted on both **in-domain** and **out-of-domain** data.

## 2. Training the Models

We utilized the following multilingual models for our experiments:

- mT5
- XLM-Roberta

### Training XLM-Roberta

To train the Claim-only Model with metadata, use the following command:

```
python train_xlmr.py --model_type claim_only --batch_size 12 --learning_rate 2e-5 --epochs 5 --max_length 360
```

To train the Attention-based Evidence Aggregator, use the following command:

```
python train_xlmr.py --model_type attn_ea --batch_size 12 --learning_rate 2e-5 --epochs 5 --max_length 360
```

### Training mT5

To train the Claim-only Model with metadata, use the following command:

```
python train_mt5.py --model_type attn_ea --batch_size 4 --learning_rate 2e-5 --epochs 5 --max_length 360
```

To train the Attention-based Evidence Aggregator, use the following command:

```
python train_mt5.py --model_type attn_ea --batch_size 4 --learning_rate 2e-5 --epochs 5 --max_length 360
```
