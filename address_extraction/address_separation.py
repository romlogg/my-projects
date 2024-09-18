import argparse
from natasha import MorphVocab
from natasha import AddrExtractor
import json

morph_vocab = MorphVocab()
addr_extractor = AddrExtractor(morph_vocab)


# Создание парсера
parser = argparse.ArgumentParser(description="separation address from text")

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
addr_dict = {}
for match in matches:
    addr_dict[match.fact.type] = match.fact.value

# Запись в файл в формате JSON
with open(json_path, 'w') as f:
    json.dump(addr_dict, f)
