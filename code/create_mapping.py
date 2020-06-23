"""
Создание маппинга заголовков в числа и обратно для каждого языка (не кросс-языковой)
"""

import argparse
from json import dump as jdump
import os
from utils.loaders import load_mapping


def parse_args(main_func):
    def wrap():
        parser = argparse.ArgumentParser(
            description='Создание маппинга индексов в заголовки и обратно и сохранение его в json')
        parser.add_argument('--texts_path', type=str, required=True,
                            help='Папка, в которой лежат тексты')
        parser.add_argument('--lang', type=str, required=True,
                            help='Язык, для которого разбираем; '
                                 'нужен для определения словаря в маппинге (ru/en')
        parser.add_argument('--mapping_path', type=str, required=True,
                            help='Файл маппинга заголовков в индексы и обратно в формате json')
        parser.add_argument('--forced', type=int, default=0,
                            help='Принудительно пересоздать весь маппинг (0|1; default: 0)')

        args = parser.parse_args()

        main_func(args)
    return wrap


@parse_args
def main(args):

    # загружаем старый, если уже был какой-то
    mapping = load_mapping(args.mapping_path, args.forced)

    i2lang = 'i2{}'.format(args.lang)
    lang2i = '{}2i'.format(args.lang)
    files = [file for file in os.listdir(args.texts_path)]
    mapping[i2lang] = {i: file for i, file in enumerate(files)}
    mapping[lang2i] = {file: i for i, file in enumerate(files)}
    print('Новый маппинг:')
    print('\t'.join(['{}: {} объекта'.format(k, len(v)) for k, v in mapping.items()]))
    jdump(mapping, open(args.mapping_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    main()
