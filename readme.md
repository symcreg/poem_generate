---
title: mengzi-t5微调古诗生成模型
author: sam
date: 2024-12-08 17:03:44
tags: 模型训练
categories: 机器学习
---
<!--more-->

### mengzi-t5预训练模型

首先在huggingface下载mengzi-t5-base模型以便后续训练。因为huggingface在国内下载速度较慢，可以使用代理下载，或者直接下载到本地再上传到服务器。这里使用[镜像网站](https://www.hf-mirror.com)下载。  
```shell
!curl -L -O https://hf-mirror.com/Langboat/mengzi-t5-base/resolve/main/pytorch_model.bin?download=True
!curl -L -O https://hf-mirror.com/Langboat/mengzi-t5-base/resolve/main/config.json?download=true
!curl -L -O https://hf-mirror.com/Langboat/mengzi-t5-base/resolve/main/spiece.vocab?download=true
!curl -L -O https://hf-mirror.com/Langboat/mengzi-t5-base/resolve/main/spiece.model?download=true
```

### 数据准备

#### 数据集下载

这里的数据是使用[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)收集的唐诗宋词，由于飞桨平台已经内置该数据集，所以我们只需添加进来就可以了，这里是解压缩数据。  
```shell
!unzip -n ./data/data70759/poems_json.zip
```

#### 数据处理

由于数据集中的诗词是繁体，使用chinese-converter库将繁体转换为简体。  
```shell
!pip install chinese-converter
```

导入库。  
```python
import json
import urllib.request
import pandas as pd
# from tqdm.notebook import tqdm
import chinese_converter  # 繁体到简体需要
import pickle
import os
import pandas as pd
import numpy as np

# IS_TEST_FLOW = True
IS_TEST_FLOW = False
```

使用IS_TEST_FLOW作为测试和训练的标志，如果是测试则只处理少量数据。  
数据集格式为json，每个json文件有1000首诗，格式如下：  
```json
[
  {
    "author": "太宗皇帝",
    "paragraphs": [
      "秦川雄帝宅，函谷壯皇居。",
      "綺殿千尋起，離宮百雉餘。",
      "連甍遙接漢，飛觀迥凌虛。",
      "雲日隱層闕，風煙出綺疎。"
    ],
    "note": [],
    "title": "帝京篇十首 一"
  }
]
```

处理json文件，创建df_list列表，每个元素是一个dataframe，最后使用pd.concat合并。  
```python
POEM_CONTENT = {
    'tang': {
        'total': 58,
        'pattern': "./poems_json/poet.tang.{0}.json"
    },
    'song': {
        'total': 255,
        'pattern': "./poems_json/poet.song.{0}.json"
    }
}

def get_poems(is_test=True, verbose=True):
  df_list = []
  for dynasty in POEM_CONTENT:
    size = 3 if is_test else POEM_CONTENT[dynasty]['total']
    for i in range(size):
      url = POEM_CONTENT[dynasty]['pattern'].format(i * 1000)
      if verbose:
        print(f"load {url} now")
      df_list.append(pd.read_json(url))
  return pd.concat(df_list)
```

使用df.apply将繁体转换为简体。  
```python
df = get_poems(is_test=IS_TEST_FLOW, verbose=True)
df['concat_paragraphs'] = [''.join(map(str, l)) for l in df['paragraphs']]
df = df[['author', 'title', 'concat_paragraphs']]

def convert_schinese(tchinese):
  return chinese_converter.to_simplified(tchinese)

df['s_content'] = df.apply(lambda row: convert_schinese(''.join(row.concat_paragraphs)), axis=1)
df['s_title'] = df.apply(lambda row: convert_schinese(''.join(row.title)), axis=1)
df['s_author'] = df.apply(lambda row: convert_schinese(''.join(row.author)), axis=1)

my_df = df
print("my_df size", len(my_df))
```

创建trim函数，替换掉一些特殊字符，限制作者、标题、内容的长度。  
```python
MAX_AUTHOR_CHAR = 4
MAX_TITLE_CHAR = 12
MIN_CONTENT_CHAR = 20
MAX_CONTENT_CHAR = 32
BAD_TOKENS = " ()[]《》（）□{}abcdefgxyz一"

def trim_author_fn(row):
  return row.s_author[:MAX_AUTHOR_CHAR]

def trim_title_fn(row):
  trimed_title = row.s_title[:MAX_TITLE_CHAR]
  for b in BAD_TOKENS:
    trimed_title = trimed_title.replace(b, "")
  return trimed_title

def trim_content_fn(row):
  trimed_content = row.s_content[:MAX_CONTENT_CHAR]
  # # End with a period to avoid partial ending to confuse model
  for b in BAD_TOKENS:
    trimed_content = trimed_content.replace(b, "")
  last_period = trimed_content.rfind("。")
  return trimed_content[:last_period+1]
  # return trimed_content

# Trim the size, a soft copy to avoid the view/copy conflict warning
my_df['s_author_trim'] = my_df.copy().apply(trim_author_fn, axis=1)
my_df['s_title_trim'] = my_df.copy().apply(trim_title_fn, axis=1)
my_df['s_content_trim'] = my_df.copy().apply(trim_content_fn, axis=1)

print("my_df size", len(my_df))
```

过滤掉一些无效数据，比如标题为空、内容太短、无正文等。  
```python
# Title cannot be empty
empty_title_mask = (my_df['s_title_trim'].str.len() == 0)
too_short_cotent_mask = (my_df['s_content_trim'].str.len() <= MIN_CONTENT_CHAR)
invalid_mask = (('无正文' == my_df['s_content_trim']) | ('无正文' == my_df['s_author_trim']))
too_short_mask =  empty_title_mask | too_short_cotent_mask | invalid_mask
# filtered_my_df = my_df.loc[too_short_mask]
# filtered_my_df

my_df = my_df.loc[~too_short_mask][[
  's_author_trim', 's_title_trim', 's_content_trim']]
print("my_df size", len(my_df))
```

```python
import re
result_dict = {
    's_author_trim': [],
    's_title_trim': [],
    's_content_trim': [],
}
for i, row in my_df.iterrows():
  c = row['s_content_trim']
  snippets = list(re.split('，|。|？', c))
  lens = [len(s) for s in snippets if s.strip() != '']
  if max(lens) != min(lens) or max(lens) not in [5, 7]:
    continue
  result_dict['s_author_trim'].append(row['s_author_trim'])
  result_dict['s_title_trim'].append(row['s_title_trim'])
  result_dict['s_content_trim'].append(c)
# print("get rid of ", sum(bad_items))
my_df = pd.DataFrame(data=result_dict)
print("left", len(my_df))
```

#### 构建数据集

构建数据集，包括source_text和target_text。  
```python
AUTHOR_PROMPT = "模仿："
TITLE_PROMPT = "作诗："
EOS_TOKEN = '</s>'
def build_dataset_df(df, include_author=True):
  dfc = df.copy()
  if include_author:
    dfc['source_text'] = TITLE_PROMPT + df['s_title_trim'] + EOS_TOKEN + AUTHOR_PROMPT + df['s_author_trim']
  else:
    dfc['source_text'] = TITLE_PROMPT + df['s_title_trim']
  dfc['target_text'] = df['s_content_trim']
  dfc = dfc[['source_text', 'target_text']]
  return dfc
```

带有作者的数据集。  
```python
df_author_title_content = build_dataset_df(my_df, True)
```

不带作者的数据集。  
```python
df_title_content = build_dataset_df(my_df, False)
```

合并数据集。  
```python
merged_df = pd.concat([df_author_title_content, df_title_content])
merged_df = merged_df.sample(frac=1.)
```
这里的frac=1.表示打乱数据集。  

### 训练

安装一下torch, simplet5等必要库。  
```shell
!pip install torch
!pip install simplet5
import torch
from simplet5 import SimpleT5
from transformers import T5Tokenizer, T5ForConditionalGeneration
```

#### 定义模型

定义模型类，继承SimpleT5，加载mengzi-t5-base模型。  
```python
torch.cuda.empty_cache()

# 指定本地模型路径
# local_model_path = "./mengzi_t5_base"
local_model_path = "./MengziT5_base"

# 定义 extra_ids 数量
extra_ids = 100

# 创建包含所有 extra_ids 的特殊标记列表
additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]


class MengziSimpleT5(SimpleT5):
  def __init__(self) -> None:
    super().__init__()
    self.device = torch.device("cuda")

  def load_my_model(self, use_gpu: bool = True):
    # self.tokenizer = T5Tokenizer.from_pretrained(local_model_path,
    # extra_ids=extra_ids,
    # additional_special_tokens=additional_special_tokens)
    self.tokenizer = T5Tokenizer.from_pretrained(local_model_path)
    self.model = T5ForConditionalGeneration.from_pretrained(local_model_path)

model = MengziSimpleT5()
model.load_my_model()
model.model = model.model.to('cuda')
```

#### 划分数据集

将数据集以0.98, 0.02的比例划分为训练集和验证集。
```python
from sklearn.model_selection import train_test_split
merged_df = merged_df.sample(frac=1) # Shuffle
train_df, eval_df = train_test_split(merged_df, test_size=0.02)
print("train", len(train_df), "eval", len(eval_df))
```

#### 训练模型

训练模型，使用train_df训练，eval_df验证。  
```python
model.train(train_df=train_df,
            eval_df=eval_df,
            source_max_token_len=(len(TITLE_PROMPT) + MAX_TITLE_CHAR +  1 + len(AUTHOR_PROMPT) + MAX_AUTHOR_CHAR),
            target_max_token_len=MAX_CONTENT_CHAR,
            batch_size=256,
            max_epochs=5,
            use_gpu=True,
            outputdir="./Models/t5-poem-v2.1")
```

#### 测试模型


使用模型生成诗词。  
```python
def poem(title_str, opt_author=None, model=model,
         is_input_traditional_chinese=False,
         num_beams=2):
  model.model = model.model.to('cuda')
  if opt_author:
    in_request = TITLE_PROMPT + title_str[:MAX_TITLE_CHAR] + EOS_TOKEN + AUTHOR_PROMPT + opt_author[:MAX_AUTHOR_CHAR]
  else:
    in_request = TITLE_PROMPT + title_str[:MAX_TITLE_CHAR]
  if is_input_traditional_chinese:
    in_request = chinese_converter.to_simplified(in_request)
  out = model.predict(in_request,
                      max_length=MAX_CONTENT_CHAR,
                      num_beams=num_beams)[0].replace(",", "，")
  if is_input_traditional_chinese:
    out = chinese_converter.to_traditional(out)
    print(f"標題： {in_request.replace('</s>', ' ')}\n詩歌： {out}")
  else:
    print(f"标题： {in_request.replace('</s>', ' ')}\n诗歌： {out}")
```

测试模型。  
```python
for title in ['秋思', "百花", '佳人有约']:
  # Empty author means general style
  for author in ['', "杜甫", "李白", "李清照", "苏轼"]:
    poem(title, author)
  print()
```

使用不同的num_beams测试模型。  
```python
for title in ['冬雪']:
  for author in  ['', "杜甫"]:
    for num_beams in (2, 3, 5, 10, 20, 50, 100, 200):
      print(f"num beams: {num_beams}")
      poem(title, author, num_beams=num_beams)
    print("-"*80)
```

### 使用模型

使用模型生成诗词。  
```python
import json
from transformers import LogitsProcessor
from transformers import LogitsProcessorList

# 具体代码
import torch
from simplet5 import SimpleT5
from transformers import T5Tokenizer, T5ForConditionalGeneration
import chinese_converter

MODEL_PATH = "./Models/t5-poem-v2.1/simplet5-epoch-4-train-loss-3.4329-val-loss-3.4315"
class PoemModel(SimpleT5):
  def __init__(self) -> None:
    super().__init__()
    self.device = torch.device("cuda")

  def load_my_model(self):
    self.tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
```

```python
# 有一些预先设定参数
AUTHOR_PROMPT = "模仿："
TITLE_PROMPT = "作诗："
EOS_TOKEN = '</s>'

poem_model = PoemModel()
poem_model.load_my_model()
poem_model.model = poem_model.model.to('cuda')

MAX_AUTHOR_CHAR = 4
MAX_TITLE_CHAR = 12
MIN_CONTENT_CHAR = 10
MAX_CONTENT_CHAR = 64
def poem(title_str, opt_author=None, model=poem_model,
         is_input_traditional_chinese=False,
         num_beams=100):
  model.model = model.model.to('cuda')
  if opt_author:
    in_request = TITLE_PROMPT + title_str[:MAX_TITLE_CHAR] + EOS_TOKEN + AUTHOR_PROMPT + opt_author[:MAX_AUTHOR_CHAR]
  else:
    in_request = TITLE_PROMPT + title_str[:MAX_TITLE_CHAR]
  if is_input_traditional_chinese:
    in_request = chinese_converter.to_simplified(in_request)
  out = model.predict(in_request,
                      max_length=MAX_CONTENT_CHAR,
                      num_beams=num_beams)[0].replace(",", "，")
                      # logits_processor=LogitsProcessorList([rhyme_processor]))[0].replace(",", "，")

  
  if is_input_traditional_chinese:
    out = chinese_converter.to_traditional(out)
    print(f"標題： {in_request.replace('</s>', ' ')}\n詩歌： {out}")
  else:
    print(f"标题： {in_request.replace('</s>', ' ')}\n诗歌： {out}")
```

```python
for title in ['秋思', '佳人', '相思',"幽梦"]:
  # Empty author means general style
  for author in ['', "杜甫", "李白", "李清照", "苏轼"]:
    poem(title, author)
  print()
```



### 结论

微调mengzi-t5模型，使用唐诗宋词数据集训练了古诗生成模型，实现古诗生成。  

slide见[这里](https://github.com/symcreg/poem_generate/blob/main/slide.pptx)。  
实现效果在[这里](https://github.com/symcreg/poem_generate/blob/main/presentation.mp4)。  
github地址：[poem_generate](https://github.com/symcreg/poem_generate)  
飞桨地址：[test](https://aistudio.baidu.com/projectdetail/8620151)  
主要参考（抄）了[chinese-ai-writing-share](https://github.com/hululuzhu/chinese-ai-writing-share)  

### 参考

0. [aistudio](https://aistudio.baidu.com/overview)
1. [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)  
2. [hf-mirror](https://hf-mirror.com)  
3. [chinese-ai-writing-share](https://github.com/hululuzhu/chinese-ai-writing-share)  
4. [aichpoem](https://github.com/wangjiezju1988/aichpoem)  
