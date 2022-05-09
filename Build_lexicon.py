import json
import time
import re

t0 = time.time()
with open('Mapping.json') as json_file:
    mapping = json.load(json_file)
print(f'Time consumed for loading mapping: {time.time() - t0} s')

t0 = time.time()
entity_start_end = []
with open('Gazetteer/Entities_recognized_all_domain.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line_in_list = line.strip().split(sep=' ')
        entity_start_end.append([mapping[line_in_list[1]], mapping[line_in_list[2]]])
del mapping
print(f'Time consumed for preparation: {time.time() - t0} s')

t0 = time.time()
# lexicon_list = ['O'] * 28149892
lexicon_list = ['O'] * 79338
for start_pos, end_pos in entity_start_end:
    for idx in range(start_pos, end_pos):
        if idx == start_pos:
            lexicon_list[idx] = 'B'
        else:
            lexicon_list[idx] = 'I'
print(f'Time consumed for building lexicon list: {time.time() - t0} s')
del entity_start_end

t0 = time.time()
pattern = re.compile(r'^\d{7}\n$')
# with open('./data_merged/train.tsv', 'r', encoding='utf-8') as f:
#     with open('./data_merged_lexicon/train.tsv', 'w', encoding='utf-8') as writer:
with open('./data_human_labeled/dev.tsv', 'r', encoding='utf-8') as f:
    with open('./data_human_labeled/dev_lexicon.tsv', 'w', encoding='utf-8') as writer:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if pattern.match(line) and lines[idx+1] == '\n' or line == "\n":
                writer.write(line)
            else:
                token = line.strip().split(sep='\t')[0].replace('\n', '')
                writer.write(token + '\t' + lexicon_list[idx] + '\n')
print(f'Time consumed for building dataset with lexicon: {time.time() - t0} s')

