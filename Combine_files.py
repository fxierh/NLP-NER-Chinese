import re
pattern = re.compile(r'^\d{7}\n$')

with open('./data_pseudo_label/train.tsv', 'r', encoding='utf-8') as f1:
    f1_lines = f1.readlines()
with open('./output_human_labeled/train_test_predictions.txt', 'r', encoding='utf-8') as f2:
    f2_lines = f2.readlines()

with open('./data_pseudo_label/train_new.tsv', 'w', encoding='utf-8') as writer:
    for idx, line in enumerate(f1_lines):
        if pattern.match(line) and f1_lines[idx+1] == '\n' or line == "\n":
            writer.write(line)
        else:
            token = line.strip().split(sep='\t')[0].replace('\n', '')
            lexicon = line.strip().split(sep='\t')[1].replace('\n', '')
            label = f2_lines[idx].strip().split(sep='\t')[-1].replace('\n', '')
            writer.write(token + '\t' + lexicon + '\t' + label + '\n')
