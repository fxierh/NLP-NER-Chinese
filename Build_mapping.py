"""
The mapping can be used for building both gazetteer and lexicon.
"""

import re
import time
import json


def count_occupancy(token):
    # token = token.lower().replace('【', '(') \
    #     .replace('（', '(').replace('】', ')').replace('）', ')').replace('，', ',').replace('“', '"') \
    #     .replace('？', '?').replace('‘', "'").replace('—', '-').replace('、', '/').replace('；', ';') \
    #     .replace('：', ':').replace('’', "'").replace('·', '.').replace('”', '"').replace('~', '-') \
    #     .replace('′', "'").replace('ⅰ', 'i').replace('》', ')').replace('《', '(').replace('ⅴ', 'v') \
    #     .replace('˙', '.').replace('×', '*').replace('ⅲ', '3').replace('ⅱ', '2').replace('ⅷ', '8') \
    #     .replace('…', '略').replace('°', '度').replace('℃', '度').replace('″', '"').replace('ⅳ', '4') \
    #     .replace('ⅵ', '6').replace('ⅶ', '7').replace('⊕', '+').replace('〈', '(').replace('〉', ')') \
    #     .replace('–', '-').replace('‖', '|').replace('．', '.').replace('ｘ', 'x').replace('ｏ', 'o') \
    #     .replace('ｉ', 'i').replace('ｈ', 'h').replace('ｓ', 's').replace('ｃ', 'c').replace('ù', 'u') \
    #     .replace('é', 'e').replace('②', '2').replace('③', '3').replace('〔', '(').replace('〕', ')') \
    #     .replace('④', '4').replace('{', '(').replace('}', ')').replace('`', "'").replace(',', '逗') \
    #     .replace('=', '等').replace('.', '点').replace('*', '乘').replace('/', '或') \
    #     .replace('α', 'A').replace('β', 'V').replace('γ', 'Y').replace('δ', 'D').replace('λ', 'L') \
    #     .replace('ε', 'E').replace('θ', 'O').replace('τ', 'T').replace('κ', 'K').replace('ω', 'W') \
    #     .replace('∞', '8').replace('μ', 'M').replace(':', '冒').replace('|', '竖').replace('ζ', 'Z') \
    #     .replace('η', 'I').replace('ν', 'v').replace('ι', 'I').replace('ξ', 'X').replace('π', 'P') \
    #     .replace('σ', 'S').replace('φ', 'F').replace('χ', 'C').replace('а', 'а').replace('∏', 'P') \
    #     .replace('∑', 'S').replace('△', 'D').replace('г', 'Y').replace('ψ', 'F').replace('∠', '角') \
    #     .replace('ρ', 'R').replace('－', '-').replace(';', '分').replace('∶', '冒').replace('∪', 'u') \
    #     .replace('(', '括').replace(')', '括')
    token = token.replace('“', '"').replace('‘', "'").replace('—', '-').replace('’', "'").replace('”', '"')\
        .replace('~', '-').replace('′', "'").replace('″', '"').replace('–', '-').replace('`', "'").replace('－', '-')
    count_variance = 0
    token_len = len(token)
    if token_len >= 1:
        count_variance -= len(re.findall('_', token))
        count_variance -= len(re.findall('"', token))
        count_variance -= len(re.findall("'", token))
        count_variance -= len(re.findall(r'\$', token))
        count_variance -= len(re.findall(r'#', token))
        count_variance -= len(re.findall(r'@', token))
        count_variance -= len(re.findall(r'¨', token))
        count_variance += len(re.findall(r'±', token))
        count_variance -= sum([len(x) - 1 for x in re.findall(r'-{2,}', token)])
        if token_len >= 3:
            count_variance -= len(re.findall(r'\\rm', token)) * 3
            count_variance -= len(re.findall(r'\\pi', token)) * 2
            if token_len >= 4:
                count_variance -= len(re.findall(r'\\tau', token)) * 3
                if token_len >= 5:
                    count_variance -= len(re.findall(r'\\left', token)) * 5
                    count_variance -= len(re.findall(r'\\circ', token)) * 5
                    count_variance -= len(re.findall(r'\\beta', token)) * 4
                    if token_len >= 6:
                        count_variance -= len(re.findall(r'\\right', token)) * 6
                        count_variance -= len(re.findall(r'\\alpha', token)) * 5
                        count_variance -= len(re.findall(r'\\gamma', token)) * 5
                        count_variance -= len(re.findall(r'\\delta', token)) * 5
                        count_variance -= len(re.findall(r'\\infty', token)) * 5
                        count_variance -= len(re.findall(r'\\kappa', token)) * 5
                        count_variance -= len(re.findall(r'\\omega', token)) * 5
                        count_variance -= len(re.findall(r'\\times', token)) * 5
                        count_variance -= len(re.findall(r'\\theta', token)) * 5
                        if token_len >= 7:
                            count_variance -= len(re.findall(r'\\lambda', token)) * 6
                            count_variance -= len(re.findall(r'\\mathbb', token)) * 7
                            count_variance -= len(re.findall(r'\\textbf', token)) * 7
                            if token_len >= 8:
                                count_variance -= len(re.findall(r'\\mathcal', token)) * 8
                                if token_len >= 11:
                                    count_variance -= len(re.findall(r'\\varepsilon', token)) * 10
    occupancy = token_len + count_variance
    assert occupancy >= 0
    return occupancy


if __name__ == "__main__":
    t0 = time.time()
    pattern = re.compile(r'^\d{7}\n$')
    map_treated_to_original = {}
    with open('./data_human_labeled/dev.tsv', 'r', encoding='utf-8') as f:
    # with open('./data_merged_gazetteer/dev.tsv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        original_index, treated_index = 0, 0
        last_token = ' '
        for idx, line in enumerate(lines):
            if pattern.match(line) and lines[idx+1] == '\n' or line == "\n":
                token = ' '
                occupancy = 1
            else:
                token = line.strip().split(sep='\t')[0]
                occupancy = count_occupancy(token)
                if token[0] in '-~—–－' and last_token[-1] in '-~—–－':
                    occupancy -= 1
                # print(token, occupancy)
            for i in range(occupancy):
                map_treated_to_original[treated_index + i] = original_index
            treated_index += occupancy
            original_index += 1
            last_token = token

    # print(map_treated_to_original)
    print(f'Time consumed for building map: {time.time() - t0} s')

    with open("Mapping.json", "w") as outfile:
        json.dump(map_treated_to_original, outfile, indent=4)
