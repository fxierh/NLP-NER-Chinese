import re
import jieba
import time

type = 'all_domain'

if __name__ == "__main__":
    pattern = re.compile(r'^\d{7}\n$')
    jieba.load_userdict(f"./Dict_with_freq_refined/{type}.txt")
    # jieba.set_dictionary(f"./Dict_with_freq_refined/{type}.txt")
    abstracts = ''

    t0 = time.time()
    with open("./data_human_labeled/dev.tsv", "r", encoding='utf-8') as f:
    # with open("./data_human_labeled/data_test_without_overlap.tsv", "r", encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if (not pattern.match(line) or lines[index+1] != '\n') and not line == "\n":
                abstracts += line.strip().split(sep='\t')[0].replace('\n', '')
            elif pattern.match(line):
                abstracts += '\n'
            else:
                abstracts += ' '
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

    # print(abstracts)
    dt = time.time() - t0
    print(f'Time consumed for rearrangement: {dt} s')

    entity = []
    with open(f"./Dict_with_freq_refined/{type}.txt", "r", encoding='utf-8') as f:
        for line in f:
            entity.append(line.strip())
    entity = set(entity)  # Membership check is faster for set/dict (hash map used) than list

    t0 = time.time()
    with open(f"./Gazetteer/Entities_recognized_{type}.txt", "w", encoding='utf-8') as writer:
        jieba.enable_parallel()
        result = jieba.tokenize(abstracts)
        for tk in result:
            word = tk[0]
            start_pos = tk[1]
            end_pos = tk[2]
            if len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', word)) - len(re.findall(r'[括逗分角竖冒或乘点等度略]', word)) > 2 \
                    and '?' not in word \
                    and word[-1] not in '-‐等的乘或_冒/' and word[0] not in '逗-冒‐+^&!>←↑.￡／%' \
                    and len(re.findall(r'[括]', word)) % 2 == 0 \
                    and not re.match(r'^[a-z]{3}$', word):
                if word in entity:
                    writer.write(word + ' ' + str(start_pos) + ' ' + str(end_pos) + '\n')
        jieba.disable_parallel()
    print(f'Time consumed for segmentation: {time.time() - t0} s')
