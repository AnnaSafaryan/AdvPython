"""
Принимаем на вход путь к папке с корпусом и путь к модели
Сохраняем json с лемматизированныеми текстами по названиям файлов
"""

import argparse
import os
from json import dump as jdump

from tqdm import tqdm

from utils.arguments import check_args
from utils.loaders import load_lemmatized, collect_texts
from utils.preprocessing import Preprocesser
from utils.required_dicts import preprocessed_required


def parse_args(main_func):
    def wrap():
        parser = argparse.ArgumentParser(
            description='Лемматизация корпуса и сохранение его в json')
        parser.add_argument('--texts_path', type=str, required=True,
                            help='Папка, в которой лежат тексты')
        parser.add_argument('--udpipe_path', type=str,
                            help='Путь к модели udpipe для обработки корпуса')
        parser.add_argument('--lemmatized_path', type=str, required=True,
                            help='Путь к файлу json, в который будут сохраняться лемматизированные файлы.'
                                 ' Если файл уже существует, он будет пополняться')
        parser.add_argument('--keep_pos', type=int, default=1,
                            help='Возвращать ли леммы, помеченные pos-тегами (0|1; default: 1)')
        parser.add_argument('--keep_stops', type=int, default=0,
                            help='Сохранять ли слова, получившие тег функциональной части речи '
                                 '(0|1; default: 0)')
        parser.add_argument('--keep_punct', type=int, default=0,
                            help='Сохранять ли знаки препинания (0|1; default: 0)')
        parser.add_argument('--forced', type=int, default=0,
                            help='Принудительно лемматизировать весь корпус заново (0|1; default: 0)')
        parser.add_argument('--preprocessed', type=int, default=0,
                            help='Лежат ли в папке уже предобработанные тексты (0|1; default: 0)')

        args = parser.parse_args()
        main_func(args)

    return wrap


def process_corpus(udpipe_path, lemmatized, texts_path, files, keep_pos, keep_punct, keep_stops):
    """
    :param udpipe_path: путь к файлу udpipe модели
    :param lemmatized: заголовки и лемматизированные тексты (словарь)
    :param texts_path: путь к папке с текстами (строка)
    :param files: заголовки текстов, которые надо лемматизировать (список строк)
    :param keep_pos: оставлять ли pos-теги (pseudo-boolean int)
    :param keep_punct: оставлять ли пунктуацию (pseudo-boolean int)
    :param keep_stops: оставлять ли стоп-слова (pseudo-boolean int)
    :return lemmatized: обновленный словарь заголовков и лемм (словарь)
    :return not_lemmatized: заголовки текстов, которые не удалось лемматизировать (список строк)
    """

    not_lemmatized = []

    preprocesser = Preprocesser(udpipe_path, keep_pos, keep_punct, keep_stops)

    for file in tqdm(files):
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').\
            read().lower().strip().splitlines()
        text_lems = preprocesser << text

        if text_lems:
            lemmatized[file] = text_lems

        else:  # с текстом что-то не так, и там не остаётся нормальных лемм
            not_lemmatized.append(file)
            continue

    return lemmatized, not_lemmatized


@parse_args
def main(args):

    # Проверяем, всё ли указали для непредобработанных статей
    check_args(args, 'preprocessed', preprocessed_required)

    lemmatized_dict = load_lemmatized(args.lemmatized_path, args.forced)

    all_files = [f for f in os.listdir(args.texts_path)]
    new_files = [file for file in all_files if file not in lemmatized_dict]
    print('Новых текстов: {}'.format(len(new_files)))

    if new_files:

        if args.preprocessed:  # если файлы уже предобработаны
            full_lemmatized_dict = collect_texts(lemmatized_dict, args.texts_path, new_files)

        else:

            full_lemmatized_dict, not_lemmatized_texts = process_corpus(
                args.udpipe_path, lemmatized_dict, args.texts_path, new_files,
                keep_pos=args.keep_pos, keep_punct=args.keep_punct, keep_stops=args.keep_stops)

            if not_lemmatized_texts:
                print('Не удалось разобрать следующие файлы:\n{}'.
                      format('\n'.join(not_lemmatized_texts)))

        jdump(full_lemmatized_dict, open(args.lemmatized_path, 'w', encoding='utf-8'))


if __name__ == "__main__":
    main()
