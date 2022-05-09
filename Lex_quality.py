import re
import json
import numpy as np

from seqeval.metrics import accuracy_score, classification_report
from seqeval.scheme import IOB2

pattern = re.compile(r'^\d{7}$')
token_types_golden_true = []
token_types_predicted = []
# with open("./data_human_labeled/train.tsv", "r", encoding='utf-8') as f:
with open("./data_pseudo_label/train.tsv", "r", encoding='utf-8') as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        if pattern.match(line) and lines[index+1] == '\n' and not line == "\n":
            token_types_golden_true.append([])
            token_types_predicted.append([])
        if not (pattern.match(line) and lines[index+1] == '\n') and not line == "\n":
            token_types_golden_true[-1].append(line.strip().split(sep='\t')[-1][0])
            token_types_predicted[-1].append(line.strip().split(sep='\t')[1])

# print(token_types_golden_true)
# print(token_types_predicted)

cls_report = classification_report(token_types_golden_true, token_types_predicted, mode='strict', scheme=IOB2, digits=4, output_dict=True)
cls_report['accuracy'] = accuracy_score(token_types_golden_true, token_types_predicted)
print(cls_report)

for key, value in cls_report.items():
    if isinstance(value, dict):
        for key2, value2 in value.items():
            if isinstance(value2, np.int64):
                cls_report[key][key2] = int(value2)
                support = value2

# with open("./data_human_labeled/lex_quality.json", "w", encoding='utf-8') as outfile:
with open("./data_pseudo_label/lex_quality.json", "w", encoding='utf-8') as outfile:
    json.dump(cls_report, outfile, ensure_ascii=False, indent=4)
