import re

# types = ['研究问题', '方法模型', '度量指标', '数据资料', '科学家', '理论原理', '仪器设备', '软件系统', '地点']
types = ['all_domain']

# chemical_elements = []
# with open("./Chemical_elements.txt", "r", encoding='utf-8') as f:
#     for line in f:
#         chemical_elements.append(line.strip())

global_dictionary = {}
for type in types:
    dictionary = {}
    with open(f"./Dict_with_freq/{type}.txt", "r", encoding='utf-8') as f:
        for i, line in enumerate(f):
            # word, freq = line.strip().split(sep=' ')
            word = line.strip()
            freq = str(1000)
            word = word.lower().replace('【', '(') \
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
                .replace('=', '等').replace('.', '点').replace('*', '乘').replace('/', '或').replace('２', '2') \
                .replace('α', 'A').replace('β', 'V').replace('γ', 'Y').replace('δ', 'D').replace('λ', 'L') \
                .replace('ε', 'E').replace('θ', 'O').replace('τ', 'T').replace('κ', 'K').replace('ω', 'W') \
                .replace('∞', '8').replace('μ', 'M').replace(':', '冒').replace('|', '竖').replace('ζ', 'Z') \
                .replace('η', 'I').replace('ν', 'v').replace('ι', 'I').replace('ξ', 'X').replace('π', 'P') \
                .replace('σ', 'S').replace('φ', 'F').replace('χ', 'C').replace('а', 'а').replace('∏', 'P') \
                .replace('∑', 'S').replace('△', 'D').replace('г', 'Y').replace('ψ', 'F').replace('∠', '角') \
                .replace('ρ', 'R').replace('－', '-').replace(';', '分').replace('∶', '冒').replace('∪', 'u') \
                .replace('(', '括').replace(')', '括')
            # Attention: string length changed. Operators to consider: (~, -, —, _, “, ”, ‘, ’, ′, etc.)
            word = re.sub(r'-{2,}', '-', word).replace('_', '').replace('"', '').replace("'", '').replace('$', '') \
                .replace(r'\left', '').replace(r'\right', '').replace('#', '').replace('@', '').replace(r'\alpha', 'A')\
                .replace(r'\beta', 'B').replace(r'\gamma', 'Y').replace(r'\delta', 'D').replace(r'\lambda', 'L') \
                .replace('¨', '').replace(r'\infty', '8').replace(r'\varepsilon', 'E').replace(r'\theta', 'O') \
                .replace(r'\rm', '').replace(r'\mathbb', '').replace(r'\tau', 'T').replace(r'\mathcal', '') \
                .replace(r'\kappa', 'K').replace(r'\omega', 'W').replace(r'\textbf', '').replace(r'\times', '乘') \
                .replace('±', '+-').replace(r'\pi', 'P').replace(r'\circ', '')

            if len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', word)) - len(re.findall(r'[括逗分角竖冒或乘点等度略]', word)) >= 2 \
                    and '?' not in word \
                    and word[-1] not in '-‐等的乘或_冒' and word[0] not in '逗-冒‐+^%':
                # len(re.findall(r'[括]', word)) % 2 == 0\
                # and not re.match(r'^[a-z]{3}$', word)\
                # and (not re.match(r'^[a-z]{4}$', word) or int(freq) > 1):
                # and (len(word) > 2 or word in chemical_elements):
                # and not re.search(r'[(][^)]*$', word) \
                # and not re.search(r'^[^(]*[)]', word) \
                # and len(re.findall(r'["]', word)) % 2 == 0
                # and len(re.findall(r'[(]', word)) == len(re.findall(r'[)]', word)) \

                if word in dictionary:
                    dictionary[word] += freq
                else:
                    dictionary[word] = freq

                if word in global_dictionary:
                    global_dictionary[word] += freq
                else:
                    global_dictionary[word] = freq

    with open(f"./Dict_with_freq_refined/{type}.txt", "w", encoding='utf-8') as writer:
        for word, freq in sorted(dictionary.items()):
            writer.write(word + ' ' + freq + '\n')
    print(f'Type {type}: {len(dictionary)} entities saved in dictionary.')

# with open(f"./Dict_with_freq_refined/所有类别.txt", "w", encoding='utf-8') as writer:
#     for word, freq in sorted(global_dictionary.items()):
#         writer.write(word + ' ' + freq + '\n')
# print(f'\nType 所有类别: {len(global_dictionary)} entities saved in dictionary.')
