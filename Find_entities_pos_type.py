import re
import jieba
import time

from jieba import posseg as pseg

type = '所有类别'

if __name__ == "__main__":
    pattern = re.compile(r'^\d{7}\n$')
    jieba.load_userdict(f"./Dict_with_freq_refined/{type}.txt")
    # jieba.set_dictionary(f"./Dict_with_freq_refined/{type}.txt")
    abstracts = ''

    t0 = time.time()
    # with open("./output_human_labeled/train_test_predictions.txt", "r", encoding='utf-8') as f:
    with open("./data_merged_gazetteer/dev.tsv", "r", encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if (not pattern.match(line) or lines[index+1] != '\n') and not line == "\n":
                abstracts += line.strip().split(sep='\t')[0]
            elif pattern.match(line):
                abstracts += '\n'
            else:
                abstracts += ' '
    # abstracts = abstracts[1:]
    abstracts = abstracts.lower().replace('【', '(') \
        .replace('（', '(').replace('】', ')').replace('）', ')').replace('，', ',').replace('“', '"') \
        .replace('？', '?').replace('‘', "'").replace('—', '-').replace('、', '/').replace('；', ';') \
        .replace('：', ':').replace('’', "'").replace('·', '.').replace('”', '"').replace('~', '-') \
        .replace('′', "'").replace('ⅰ', 'i').replace('》', ')').replace('《', '(').replace('ⅴ', 'v') \
        .replace('˙', '.').replace('×', '*').replace('ⅲ', '3').replace('ⅱ', '2').replace('ⅷ', '8') \
        .replace('…', '略').replace('°', '度').replace('℃', '度').replace('″', '"').replace('ⅳ', '4') \
        .replace('ⅵ', '6').replace('ⅶ', '7').replace('⊕', '+').replace('〈', '(').replace('〉', ')') \
        .replace('–', '-').replace('‖', '|').replace('．', '.').replace('ｘ', 'x').replace('ｏ', 'o') \
        .replace('ｉ', 'i').replace('ｈ', 'h').replace('ｓ', 's').replace('ｃ', 'c').replace('ù', 'u') \
        .replace('é', 'e').replace('②', '2').replace('③', '3').replace('〔', '(').replace('〕', ')') \
        .replace('④', '4').replace('{', '(').replace('}', ')').replace('`', "'").replace(',', '逗') \
        .replace('=', '等').replace('.', '点').replace('*', '乘').replace('/', '或') \
        .replace('α', 'A').replace('β', 'V').replace('γ', 'Y').replace('δ', 'D').replace('λ', 'L') \
        .replace('ε', 'E').replace('θ', 'O').replace('τ', 'T').replace('κ', 'K').replace('ω', 'W') \
        .replace('∞', '8').replace('μ', 'M').replace(':', '冒').replace('|', '竖').replace('ζ', 'Z') \
        .replace('η', 'I').replace('ν', 'v').replace('ι', 'I').replace('ξ', 'X').replace('π', 'P') \
        .replace('σ', 'S').replace('φ', 'F').replace('χ', 'C').replace('а', 'а').replace('∏', 'P') \
        .replace('∑', 'S').replace('△', 'D').replace('г', 'Y').replace('ψ', 'F').replace('∠', '角') \
        .replace('ρ', 'R').replace('－', '-').replace(';', '分').replace('∶', '冒').replace('∪', 'u') \
        .replace('(', '括').replace(')', '括')
    # Attention: string length changed. Operators to consider: (~, -, —, _, “, ”, ‘, ’, ′, etc.)
    abstracts = re.sub(r'-{2,}', '-', abstracts).replace('_', '').replace('"', '').replace("'", '').replace('$', '') \
        .replace(r'\left', '').replace(r'\right', '').replace('#', '').replace('@', '').replace(r'\alpha', 'A') \
        .replace(r'\beta', 'B').replace(r'\gamma', 'Y').replace(r'\delta', 'D').replace(r'\lambda', 'L') \
        .replace('¨', '').replace(r'\infty', '8').replace(r'\varepsilon', 'E').replace(r'\theta', 'O') \
        .replace(r'\rm', '').replace(r'\mathbb', '').replace(r'\tau', 'T').replace(r'\mathcal', '') \
        .replace(r'\kappa', 'K').replace(r'\omega', 'W').replace(r'\textbf', '').replace(r'\times', '乘') \
        .replace('±', '+-').replace(r'\pi', 'P').replace(r'\circ', '')

    print(abstracts)
    dt = time.time() - t0
    print(f'Time consumed for rearrangement: {dt} s')

    with open("./Dict_with_freq_refined/研究问题.txt", "r", encoding='utf-8') as f:
        entities_1 = {}
        for line in f:
            entities_1[line.strip().split(sep=' ')[0]] = int(line.strip().split(sep=' ')[1])

    with open("./Dict_with_freq_refined/方法模型.txt", "r", encoding='utf-8') as f:
        entities_2 = {}
        for line in f:
            entities_2[line.strip().split(sep=' ')[0]] = int(line.strip().split(sep=' ')[1])

    with open("./Dict_with_freq_refined/度量指标.txt", "r", encoding='utf-8') as f:
        entities_3 = {}
        for line in f:
            entities_3[line.strip().split(sep=' ')[0]] = int(line.strip().split(sep=' ')[1])

    with open("./Dict_with_freq_refined/数据资料.txt", "r", encoding='utf-8') as f:
        entities_4 = {}
        for line in f:
            entities_4[line.strip().split(sep=' ')[0]] = int(line.strip().split(sep=' ')[1])

    with open("./Dict_with_freq_refined/科学家.txt", "r", encoding='utf-8') as f:
        entities_5 = {}
        for line in f:
            entities_5[line.strip().split(sep=' ')[0]] = int(line.strip().split(sep=' ')[1])

    with open("./Dict_with_freq_refined/理论原理.txt", "r", encoding='utf-8') as f:
        entities_6 = {}
        for line in f:
            entities_6[line.strip().split(sep=' ')[0]] = int(line.strip().split(sep=' ')[1])

    with open("./Dict_with_freq_refined/仪器设备.txt", "r", encoding='utf-8') as f:
        entities_7 = {}
        for line in f:
            entities_7[line.strip().split(sep=' ')[0]] = int(line.strip().split(sep=' ')[1])

    with open("./Dict_with_freq_refined/软件系统.txt", "r", encoding='utf-8') as f:
        entities_8 = {}
        for line in f:
            entities_8[line.strip().split(sep=' ')[0]] = int(line.strip().split(sep=' ')[1])

    with open("./Dict_with_freq_refined/地点.txt", "r", encoding='utf-8') as f:
        entities_9 = {}
        for line in f:
            entities_9[line.strip().split(sep=' ')[0]] = int(line.strip().split(sep=' ')[1])

    t0 = time.time()
    with open("./Gazetteer/Entities_recognized.txt", "w", encoding='utf-8') as writer:
        jieba.enable_parallel()
        # words_pseg = pseg.cut(abstracts)
        # for word, flag in words_pseg:
        #     if flag == 'x' or flag == 'n':
        #         entities.append([word, flag])
        result = jieba.tokenize(abstracts)
        for tk in result:
            word = tk[0]
            start_pos = tk[1]
            end_pos = tk[2]
            if len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', word)) - len(re.findall(r'[括逗分角竖冒或乘点等度略]', word)) > 2 \
                    and '?' not in word \
                    and word[-1] not in '-‐等的乘或_冒' and word[0] not in '逗-冒‐+^' \
                    and len(re.findall(r'[括]', word)) % 2 == 0 \
                    and not re.match(r'^[a-z]{3}$', word):
                word_type = 0
                max_freq = 0
                if word in entities_1:
                    word_type = '研究问题'
                    max_freq = entities_1[word]
                if word in entities_2 and entities_2[word] > max_freq:
                    word_type = '方法模型'
                    max_freq = entities_2[word]
                if word in entities_3 and entities_3[word] > max_freq:
                    word_type = '度量指标'
                    max_freq = entities_3[word]
                if word in entities_4 and entities_4[word] > max_freq:
                    word_type = '数据资料'
                    max_freq = entities_4[word]
                if word in entities_5 and entities_5[word] > max_freq:
                    word_type = '科学家'
                    max_freq = entities_5[word]
                if word in entities_6 and entities_6[word] > max_freq:
                    word_type = '理论原理'
                    max_freq = entities_6[word]
                if word in entities_7 and entities_7[word] > max_freq:
                    word_type = '仪器设备'
                    max_freq = entities_7[word]
                if word in entities_8 and entities_8[word] > max_freq:
                    word_type = '软件系统'
                    max_freq = entities_8[word]
                if word in entities_9 and entities_9[word] > max_freq:
                    word_type = '地点'
                    max_freq = entities_9[word]

                if word_type:
                    writer.write(word + ' ' + str(start_pos) + ' ' + str(end_pos) + ' ' + str(word_type) + ' ' + str(max_freq) + '\n')
        jieba.disable_parallel()
    print(f'Time consumed for segmentation: {time.time() - t0} s')
