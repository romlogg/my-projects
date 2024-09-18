import argparse
from natasha import MorphVocab
from natasha import AddrExtractor
import json

morph_vocab = MorphVocab()
addr_extractor = AddrExtractor(morph_vocab)


# Создание парсера
parser = argparse.ArgumentParser(description="extracting address from text")

# Добавление аргументов
parser.add_argument("-in_file", dest="txt_path", required=True, help='путь к файлу txt')
parser.add_argument("-out_file", dest="json_path", required=True, help="путь для сохранения фала json")
args = parser.parse_args()
txt_path = args.txt_path
json_path = args.json_path

# Открыть файл txt
with open(txt_path, 'r', encoding='utf-8') as f:
    text = f.read()

matches = addr_extractor(text)

start = []
stop = []
for match in matches:
    strt =  match.start
    stp =  match.stop
    if not start:
        start.append(strt)
        stop.append(stp)
    if strt - stop[-1] > 10:
        start.append(strt)
        stop.append(stp)
    stop[-1]=stp

addr_list = []
for i in range(len(start)):
    addr_list.append(text[start[i]:stop[i]])

# Запись в файл в формате JSON
with open(json_path, 'w') as f:
    json.dump(addr_list, f)
