# DaSE-Comtemporary-AI-V

This repository is a contemporary artificial intelligence experiment five repository. This experiment is mainly a multimodal fusion experiment, and set up a corresponding baseline for the ablation experiment comparison process.

## Requirements

- chardet==3.0.4  

- numpy==1.19.2  

- pandas==1.1.3  

- scikit_learn==1.1.1  

- torch==1.10.2  

- torchvision==0.11.3  

- tqdm==4.64.0  

- transformers==4.19.2

You can simply run

```shell
pip install -r requirements.txt
```

## Repository structure

We select some important files for detailed description.

```shell
├────attention_model.py
├────data/
│    └────实验五要求.pptx
├────main.py
├────model.py
├────picture_only/
│    ├────main.py
│    ├────model.py
│    ├────read_picture.py
│    └────train.py
├────requirements.txt
├────text_only/
│    ├────bert.py
│    ├────main.py
│    ├────out.txt
│    ├────read_txt.py
│    └────train.py
├────train.py
└────utils.py
```

To import the baseline bert model

```shell
git lfs install
git clone https://huggingface.co/bert-base-multilingual-cased
```

## Dataset

To fetch the dataset you can contact me.

And the dataset is made up of 4000 training dataset. And each element in this dataset is composed of one picture, one text and a label. Our task is to predict the emotion of these pictures and text.

## Train the model

1. You can run any models implemented in 'models.py'. For examples, you can run model 'sum' with:
   
   ```python
   python main.py --model sum --lr 5e-5
   ```
   
   And you can run other models, such as
   
   ```python
   python main.py --model concat --lr 5e-5
   ```


