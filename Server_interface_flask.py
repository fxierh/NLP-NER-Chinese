#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import re
from embed.models.transformers import BertForTokenClassification, BertTokenizer

from tqdm import tqdm
import numpy as np
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_list, get_lexicon, get_pos
from tst import text_to_word_list, split_zh_en, whole_text_connection, remove_space_by_slash_and_Hyphen, C_trans_to_E
from collections import Counter

import os
import sys

os.chdir(sys.path[0])

import time
import logging

logging.basicConfig(level=logging.INFO)

from flask import Flask, request, render_template

app = Flask(__name__, template_folder='Templates')

# Some Parameters
# output_dir = r'output'
output_dir = r'output_pseudo_label'
MAX_SEQ_LENGTH = 512
MODEL_TYPE = 'bert'
EVAL_BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-Load model
tokenizer = BertTokenizer.from_pretrained('./Tokenizer/')
model = BertForTokenClassification.from_pretrained(output_dir)
model.to(DEVICE)

# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index

labels = get_labels(path='labels.txt')
pos = get_pos(None)
lexicon = get_lexicon(path='lexicon_map.txt')

# 读取化学元素
with open(r'化学元素.txt', 'r', encoding='utf-8') as f:
    elements = f.readlines()
elements = [ele.strip() for ele in elements]


def load_and_cache_chinese_examples(text):
    global word_list
    word_list = text_to_word_list(text)  ###word_list是[(字，标签)]的格式
    print(word_list)
    # print(len(word_list))

    examples = read_examples_from_list(word_list)
    features = convert_examples_to_features(examples, labels, pos, lexicon, MAX_SEQ_LENGTH, tokenizer,
                                            cls_token_at_end=bool(MODEL_TYPE in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if MODEL_TYPE in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(MODEL_TYPE in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(MODEL_TYPE in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if MODEL_TYPE in ["xlnet"] else 0,
                                            pad_token_label_id=pad_token_label_id)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lexicon_ids = torch.tensor([f.lexicon_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pos_ids, all_label_ids, all_lexicon_ids)
    return dataset


def predict_keywords_bio_lexicon(text):
    eval_dataset = load_and_cache_chinese_examples(text)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[4]}
            # "lexicon_ids": batch[5]}
            if MODEL_TYPE != "distilbert":
                inputs["token_type_ids"] = batch[2] if MODEL_TYPE in ["bert",
                                                                      "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)
    # print(preds)
    # print(len(preds[0]))

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    # print(out_label_list)
    # print(len(out_label_list[0]))
    # print(preds_list)
    # print(len(preds_list[0]))

    # word_list = ['目', '的', ' ', '探', '讨', '经', '皮', '内', '镜', '椎', '间', '孔', '入', '路', '微', '创', '治', '疗']
    # label_list = ['O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'B', 'I', 'I', 'I', 'I', 'B', 'I', 'O', 'O']
    global word_list
    word_list = [i[0] for i in word_list[:len(preds_list[0])]]  ##第二个元素是标签
    label_list = preds_list[0]  ##已经将pad的去掉了
    # print(list(zip(word_list,label_list)))
    all_keys = []
    for i, token in enumerate(word_list):
        if label_list[i].startswith('B'):
            if i == len(word_list) - 1 and (split_zh_en(token)[0][0] == 1 or token in elements):
                all_keys.append([token, label_list[i].split('-')[-1].strip()])
            if i != len(word_list) - 1 and label_list[i + 1].startswith('O'):
                if split_zh_en(token)[0][0] == 1 or token in elements:
                    all_keys.append([token, label_list[i].split('-')[-1].strip()])
                else:
                    continue
            if i != len(word_list) - 1 and label_list[i + 1].startswith('I'):
                keys = [token]
                all_labels = [label_list[i].split('-')[-1].strip()]
                for z in range(i + 1, len(word_list)):
                    if label_list[z].startswith('I'):
                        keys.append(word_list[z])
                        all_labels.append(label_list[z].split('-')[-1].strip())
                    else:
                        break
                if len(list(set(all_labels))) == 1:
                    extract_key = whole_text_connection(keys)
                    if '和' == extract_key[-1] or '的' == extract_key[-1]:
                        pass
                    ## todo 直接删除缺失括号的
                    # elif extract_key.count("）")!=extract_key.count("（") and (extract_key.count("）")!=0 or extract_key.count("（")!=0):
                    #     pass
                    elif extract_key.count('）') > extract_key.count('（') and len(extract_key) > 1:
                        new_extract_key = "（" + extract_key
                        all_keys.append([new_extract_key, all_labels[0]])
                    elif extract_key.count('）') < extract_key.count('（') and len(extract_key) > 1:
                        new_extract_key = extract_key + '）'
                        all_keys.append([new_extract_key, all_labels[0]])
                    elif extract_key.count('】') > extract_key.count('【') and len(extract_key) > 1:
                        new_extract_key = "【" + extract_key
                        all_keys.append([new_extract_key, all_labels[0]])
                    elif extract_key.count('】') < extract_key.count('【') and len(extract_key) > 1:
                        new_extract_key = extract_key + '】'
                        all_keys.append([new_extract_key, all_labels[0]])
                    else:
                        all_keys.append([extract_key, all_labels[0]])

    ## todo 进行去重，对于列表的去重，使用Counter()
    all_keyphrase_str = [str(ele) for ele in all_keys]
    new_all_keys = [eval(ele) for ele in Counter(all_keyphrase_str).keys()]

    #####################todo 将具有包含关系的去掉
    del_ele = []
    for i, keyphrase in enumerate(new_all_keys):
        for z in range(len(new_all_keys)):
            if keyphrase[0] in new_all_keys[z][0] and z != i and keyphrase[1] == new_all_keys[z][
                1]:  ## todo 类别相同的具有包含关系的才删除
                del_ele.append(keyphrase)
    del_ = []
    for zk in del_ele:
        if zk not in del_:  ## todo 不能直接用set，因为元素是列表
            del_.append(zk)
    if len(del_) != 0:
        for ele in del_:
            new_all_keys.remove(ele)

    ## todo 去掉以特殊字符结尾的
    pattern_end = re.compile(r'(.*)[_:∶！。，（；、？——+=]$')
    new_article_keys = []
    for i in new_all_keys:
        if pattern_end.match(i[0]) != None:
            i = [i[0][:-1], i[1]]
            new_article_keys.append(i)
        else:
            new_article_keys.append(i)

    return new_article_keys


def entity_type_combination(article_keys):
    question = []
    method = []
    metric = []
    dataset = []
    scientist = []
    theory = []
    equipment = []
    software = []
    location = []
    all = []
    for ele in article_keys:
        if ele[1] == '研究问题':
            question.append(ele[0])
        if ele[1] == '方法模型':
            method.append(ele[0])
        if ele[1] == '度量指标':
            metric.append(ele[0])
        if ele[1] == '数据资料':
            dataset.append(ele[0])
        if ele[1] == '科学家':
            scientist.append(ele[0])
        if ele[1] == '理论原理':
            theory.append(ele[0])
        if ele[1] == '仪器设备':
            equipment.append(ele[0])
        if ele[1] == '软件系统':
            software.append(ele[0])
        if ele[1] == '地点':
            location.append(ele[0])
    all.append([question, method, metric, dataset, scientist, theory, equipment, software, location])
    return all


@app.route('/', methods=['POST', 'GET'])
def enter_abstract():
    return render_template('form.html', form=enter_abstract)


@app.route('/results', methods=['POST'])
def NER_SCI_CN():
    if request.method == 'POST':
        text = request.form["Abstract"]
    # text = request.form["text"]
    print(text)

    result_return = {}
    processed_text = C_trans_to_E(remove_space_by_slash_and_Hyphen(text))
    start = time.time()
    all_keys = predict_keywords_bio_lexicon(text)
    all_entities = entity_type_combination(all_keys)[0]
    new_all_entities = [[], [], [], [], [], [], [], [], []]
    for i, kk in enumerate(all_entities):
        for entity_type in enumerate(kk):
            new_entity = C_trans_to_E(entity_type[1].lower())
            try:
                start_loc = processed_text.lower().index(new_entity)
                end_loc = start_loc + len(entity_type[1])
                new_all_entities[i].append(processed_text[start_loc:end_loc])
            except:
                new_all_entities[i].append(new_entity)
                print('报错啦!!!!!!{}'.format(entity_type))

    for i, ele in enumerate(new_all_entities):
        if i == 0 and len(ele) != 0:
            print('研究问题：{}'.format('                    '.join(ele)))
            result_return['研究问题'] = ele

        if i == 1 and len(ele) != 0:
            print('方法模型：{}'.format('                    '.join(ele)))
            result_return['方法模型'] = ele

        if i == 2 and len(ele) != 0:
            print('度量指标：{}'.format('                    '.join(ele)))
            result_return['度量指标'] = ele

        if i == 3 and len(ele) != 0:
            print('数据资料：{}'.format('                    '.join(ele)))
            result_return['数据资料'] = ele

        if i == 4 and len(ele) != 0:
            print('科学家：{}'.format('                    '.join(ele)))
            result_return['科学家'] = ele

        if i == 5 and len(ele) != 0:
            print('理论原理：{}'.format('                    '.join(ele)))
            result_return['理论原理'] = ele

        if i == 6 and len(ele) != 0:
            print('仪器设备：{}'.format('                    '.join(ele)))
            result_return['仪器设备'] = ele

        if i == 7 and len(ele) != 0:
            print('软件系统：{}'.format('                    '.join(ele)))
            result_return['软件系统'] = ele

        if i == 8 and len(ele) != 0:
            print('地点：{}'.format('                    '.join(ele)))
            result_return['地点'] = ele
    end = time.time()
    print('用时:{}秒'.format(end - start))
    # print(result_return)

    # return result_return

    return render_template('data.html',
                           result=result_return,
                           time_consumed=end - start, device=DEVICE)


def predict(text):
    result_return = {}
    processed_text = C_trans_to_E(remove_space_by_slash_and_Hyphen(text))
    start = time.time()
    all_keys = predict_keywords_bio_lexicon(text)
    all_entities = entity_type_combination(all_keys)[0]
    new_all_entities = [[], [], [], [], [], [], [], [], []]
    for i, kk in enumerate(all_entities):
        for entity_type in enumerate(kk):
            new_entity = C_trans_to_E(entity_type[1].lower())
            try:
                start_loc = processed_text.lower().index(new_entity)
                end_loc = start_loc + len(entity_type[1])
                new_all_entities[i].append(processed_text[start_loc:end_loc])
            except:
                new_all_entities[i].append(new_entity)
                print('报错啦!!!!!!{}'.format(entity_type))

    for i, ele in enumerate(new_all_entities):
        if i == 0 and len(ele) != 0:
            result_return['研究问题'] = ele

        if i == 1 and len(ele) != 0:
            result_return['方法模型'] = ele

        if i == 2 and len(ele) != 0:
            result_return['度量指标'] = ele

        if i == 3 and len(ele) != 0:
            result_return['数据资料'] = ele

        if i == 4 and len(ele) != 0:
            result_return['科学家'] = ele

        if i == 5 and len(ele) != 0:
            result_return['理论原理'] = ele

        if i == 6 and len(ele) != 0:
            result_return['仪器设备'] = ele

        if i == 7 and len(ele) != 0:
            result_return['软件系统'] = ele

        if i == 8 and len(ele) != 0:
            result_return['地点'] = ele
    end = time.time()
    print('用时:{}秒'.format(end - start))
    print(result_return)
    return result_return


if __name__ == '__main__':
    # app.run('0.0.0.0', port=7038, debug=False)
    # app.run(host='localhost', debug=True)
    text = "基于GC-MS指纹图谱结合化学计量学的不同产地香附挥发油化学成分的比较研究。目的比较不同产地香附挥发油化学成分的种类与含量，为香附种质筛选与开发利用提供参考。方法采用GC-MS技术，对不同产地的香附药材挥发油成分进行定性分析；建立香附药材挥发油成分指纹图谱并进行相似度评价，结合聚类分析（HCA）、主成分分析（PCA）等化学计量学方法比较不同产地香附挥发油化学成分种类差异；通过HPLC法对香附挥发油中的有效成分（香附烯酮、α-香附酮、圆柚桐）进行定量分析，比较不同产地样品的含量差异。结果从12个产地香附挥发油中共鉴定出46种不同化合物，通过峰面积归一化法计算各成分相对百分含量，其中香附子烯（18.87％）的相对含量较高；通过建立香附挥发油GC-MS指纹图谱，发现12个产地样品中含有19个共有峰，并鉴定其化学名称；12个产地香附挥发油成分的相似度平均值为0.91；聚类分析结果显示，12个产地的香附挥发油样品可聚成3类；主成分分析结果显示前4个主成分的累计贡献率为97.621％，可以反映原始色谱峰的大部分信息，综合得分结果显示云南、山西产香附相比其他产地质量更好；对三种有效成分的含量测定结果显示，香附烯酮、圆柚酮以及α-香附酮的含量在不同产地存在显著性差异。结论香附在我国分布范围广，不同产地香附挥发油化学成分种类与含量均存在一定差异，本研究通过对香附挥发油化学成分的分析，以期为香附挥发油质量控制与产品开发提供理论依据。"
    predict(text)
