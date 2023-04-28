# LEWIT-informal

This repository presents LEWIT-informal model, designed to transfer formal text into informal keeping the important slots from the source text. The slots can be either pre-defined or detected automatically.

The model is based on LEWIT (T5-based LEWIS). It exploits the ability of T5 model to fill the slots between the known texts with undefined number of tokens. 

## How to use

### Installation

```python
!pip install lewip_informal
```

### Generating content preserving informal paraphrases

#### When the important entities are known in advance

```python
from lewip_informal import LEWIP
model = LEWIP(use_cuda = True)
text = 'I want to go to NY'
ent = ['NY']
model.generate(text, ent)
# expected output 'i wanna go to NY'
```

#### When the important entities are NOT known in advance

In case the important slots are not known, they are automatically detected with auxiliary tagger model. 

```python
from lewip_informal import LEWIP
model = LEWIP(predefined_entities = False, use_cuda = True)
text = 'I really want to travel to NY'
model.generate(text)
# expected output 'I really want to go to NY'
```

You may want to have a look at the slots which were labeled as important by the tagger. Use 'show_template' variable

```python

model.generate(text, show_template = True)
# expected output 
# I <extra_id_0> want to <extra_id_1> to NY
# 'I really want to go to NY'

```

# Dataset

The dataset used for evaluation of content preserving formality transfer. Learn more [here](dataset/README.md)
