import json
import time
import re

t0 = time.time()
with open('Mapping.json') as json_file:
    mapping = json.load(json_file)
print(f'Time consumed for loading mapping: {time.time() - t0} s')

t0 = time.time()
entity_start_end_type = []
with open('Gazetteer/Entities_recognized.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line_in_list = line.strip().split(sep=' ')
        entity_start_end_type.append((mapping[line_in_list[1]], mapping[line_in_list[2]], line_in_list[3]))
del mapping
print(f'Time consumed for preparation: {time.time() - t0} s')

t0 = time.time()
# gazetteer_list = ['O']*30198178
gazetteer_list = ['O']*24509
for start_pos, end_pos, entity_type in entity_start_end_type:
    for idx in range(start_pos, end_pos):
        if idx == start_pos:
            gazetteer_list[idx] = 'B-' + entity_type
        else:
            gazetteer_list[idx] = 'I-' + entity_type
print(f'Time consumed for building gazetteer list: {time.time() - t0} s')
del entity_start_end_type

t0 = time.time()
pattern = re.compile(r'^\d{7}\n$')
# with open('./output_human_labeled/train_test_predictions.txt', 'r', encoding='utf-8') as f:
#     with open('./data_merged_gazetteer/train.tsv', 'w', encoding='utf-8') as writer:
with open('./data_merged_gazetteer/dev.tsv', 'r', encoding='utf-8') as f:
    with open('./data_merged_gazetteer/dev_2.tsv', 'w', encoding='utf-8') as writer:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if pattern.match(line) and lines[idx+1] == '\n' or line == "\n":
                writer.write(line)
            else:
                token = line.strip().split(sep='\t')[0]
                label = line.strip().split(sep='\t')[-1]
                writer.write(token + '\t' + gazetteer_list[idx] + '\t' + label + '\n')
print(f'Time consumed for building gazetteer dataset: {time.time() - t0} s')

