import re

types = ['研究问题', '方法模型', '度量指标', '数据资料', '科学家', '理论原理', '仪器设备', '软件系统', '地点']
pattern = re.compile(r'^\d{7}\n$')

single_character_list = []
with open("./化学元素.txt", "r", encoding='utf-8') as f:
    for line in f:
        single_character_list.append(line.strip()[0])

for tp in types:
    token_types = []
    tokens = []
    with open("./output_human_labeled/train_test_predictions.txt", "r", encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if (not pattern.match(line) or lines[index+1] != '\n') and not line == "\n":
                if line.strip().split(sep='\t')[-1][-len(tp):] == tp:
                    token_types.append(line.strip().split(sep='\t')[-1][0])
                else:
                    token_types.append('O')
                tokens.append(line.strip().split(sep='\t')[0])
            elif pattern.match(line):
                token_types.append(line.strip()[0])
                tokens.append(' ')
            else:
                token_types.append(' ')
                tokens.append(' ')

    token_types = "".join(token_types)  # Ex: "6 OOOOOOOBIIIIIIIIOBIIIIOOOOOOOOOOOOBIIIIIIIIIIIIIIIIOBIIOOOOOOOOOOB..."
    token_span = [(m.start(0), m.end(0)) for m in re.finditer(r'BI*', token_types)]
    all_entities = ["".join(tokens[start_pos:end_pos]) for (start_pos, end_pos) in token_span]
    unique_entities = list(set(["".join(tokens[start_pos:end_pos]) for (start_pos, end_pos) in token_span]))

    print(f'Type "{tp}": {len(all_entities)} entities in total, {len(unique_entities)} unique entities in total.')

    freq = {}  # Frequency of each unique entity
    for item in all_entities:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1

    with open(f"./Dict_with_freq/{tp}.txt", "w", encoding='utf-8') as writer:
        for key, value in freq.items():
            if len(key.strip()) > 1 or key.strip() in single_character_list:
                writer.write(key + ' ' + str(value) + '\n')
