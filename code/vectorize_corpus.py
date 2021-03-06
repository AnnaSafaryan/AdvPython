"""
Пайплайн для одноязычной и кросс-языковой векторизации корпусов.
Для кросс-языковой векторизации собирает общий маппинг и сохраняет в pkl общую маторицу векторов
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from json import load as jload, dump as jdump
from pickle import dump as pdump

from utils.arguments import check_args
from utils.errors import NotLemmatizedError
from utils.loaders import get_lemmatized_corpus, load_vectorized
from utils.required_dicts import lang_required, cross_method_required, mono_method_required
from utils.vectorization import build_vectorizer


def parse_args(main_func):
    def wrap():
        parser = argparse.ArgumentParser(
            description='Пайплайн для векторизации двух корпусов любой моделью '
                        'и составления общей матрицы и общего маппинга')
        parser.add_argument('--src_lemmatized_path', type=str, required=True,
                            help='Путь к лемматизованным текстам на исходном языке в формате json')
        parser.add_argument('--tar_lemmatized_path', type=str, required=True,
                            help='Путь к лемматизованным текстам на целевом языке в формате json')
        parser.add_argument('--lang', type=str, default='cross',
                            help='Язык, для которого векторизуем; '
                                 'нужен для определения словаря в маппинге (ru/en/cross, default: cross')
        parser.add_argument('--direction', type=str,
                            help='Направление перевода векторов при кросс-языковой векторизации (ru-en)')
        parser.add_argument('--method', type=str, required=True,
                            help='Метод векторизации (model/translation/projection)')
        parser.add_argument('--mapping_path', type=str, required=True,
                            help='Файл маппинга заголовков в индексы и обратно в формате json')
        parser.add_argument('--src_embeddings_path', type=str,
                            help='Путь к модели векторизации для исходного языка')
        parser.add_argument('--tar_embeddings_path', type=str,
                            help='Путь к модели векторизации для целевого языка')
        parser.add_argument('--common_output_vectors_path', type=str,
                            help='Путь к pkl, в котором лежит объединённый векторизованный корпус')
        parser.add_argument('--src_output_vectors_path', type=str,
                            help='Путь к pkl, в котором лежит '
                                 'уже векторизованный корпус на исходном языке')
        parser.add_argument('--tar_output_vectors_path', type=str,
                            help='Путь к pkl, в котором лежит '
                                 'уже векторизованный корпус на целевом языке')
        parser.add_argument('--bidict_path', type=str,
                            help='Путь к двуязычному словарю в формате txt')
        parser.add_argument('--projection_path', type=str,
                            help='Путь к матрице трансформации в формате txt')
        parser.add_argument('--no_duplicates', type=int, default=0,
                            help='Брать ли для каждого типа в тексте вектор только по одному разу '
                                 '(0|1; default: 0)')
        parser.add_argument('--forced', type=int, default=0,
                            help='Принудительно векторизовать весь корпус заново (0|1; default: 0)')

        args = parser.parse_args()
        main_func(args)

    return wrap


def vectorize_corpus(corpus, vectors, vectorizer, starts_from=0):
    """векторизация всего корпуса. Если матрицей, то в model будет матрица трансформаци"""
    not_vectorized = []

    for i, text in tqdm(enumerate(corpus)):
        vector = vectorizer << text
        # print(vector.shape)
        # print(vectorizer.empty.shape)
        if not np.array_equal(vector, vectorizer.empty):  # если это не пустой вектор
            vectors[starts_from + i, :] = vector[:]  # дописывам вектора новых текстов в конец

        else:
            not_vectorized.append(str(i))
            continue

    return vectors, not_vectorized


def main_onelang(direction, lang, texts_mapping, lemmatized_path, embeddings_path,
                 output_vectors_path, method, no_duplicates, projection_path, bidict_path, forced):
    i2lang = 'i2{}'.format(lang)

    # собираем лемматизированные тексты из lemmatized
    if not os.path.isfile(lemmatized_path):  # ничего ещё из этого корпуса не разбирали
        raise NotLemmatizedError()

    else:  # если существует уже разбор каких-то файлов
        lemmatized_dict = jload(open(lemmatized_path, encoding='utf-8'))
        print('Понял, сейчас векторизуем.')

        # подгружаем старое, если было
        old_vectorized = load_vectorized(output_vectors_path, forced)

        # появились ли новые номера в маппинге
        n_new_texts = len(texts_mapping[i2lang]) - len(old_vectorized)
        print('Новых текстов: {}'.format(n_new_texts))

        if not n_new_texts:  # если не нашлось новых текстов
            return old_vectorized

        else:
            # собираем всё, что есть лемматизированного и нелемматизированного
            lemmatized_corpus = get_lemmatized_corpus(texts_mapping, i2lang, lemmatized_dict,
                                                      n_new_texts)
            # for i in lemmatized_corpus:
            #     print(i)

            # для tar всегда загружаем верисю model
            vectorizer = build_vectorizer(direction, method, embeddings_path, no_duplicates,
                                          projection_path, bidict_path)

            # за размер нового корпуса принимаем длину маппинга
            new_vectorized = np.zeros((len(texts_mapping[i2lang]), vectorizer.dim))

            # заполняем старые строчки, если они были
            for nr, line in enumerate(old_vectorized):
                new_vectorized[nr, :] = line
            # print(new_vectorized)
            # print(new_vectorized.shape)

            new_vectorized, not_vectorized = vectorize_corpus(
                lemmatized_corpus, new_vectorized, vectorizer, starts_from=len(old_vectorized))

            if output_vectors_path:
                pdump(new_vectorized, open(output_vectors_path, 'wb'))

            if not_vectorized:
                print('Не удалось векторизовать следующие тексты:\n{}'.
                      format('\t'.join(not_vectorized)))

            return new_vectorized



def to_common(texts_mapping, common2i, i2common, common_vectorized, vectorized, lang, start_from=0):
    '''добавляем корпус и заголовки в общую матрицу и общий маппинг'''
    for nr in range(len(vectorized)):
        common_vectorized[nr + start_from, :] = vectorized[nr]

        title = texts_mapping['i2{}'.format(lang)][str(nr)]
        common2i[title] = nr + start_from
        i2common[nr + start_from] = title

    return common_vectorized, common2i, i2common


@parse_args
def main(args):

    texts_mapping = jload(open(args.mapping_path))

    # для кросс-языковой векторизации должно быть указано направление и путь к общей матрице векторов
    check_args(args, 'lang', lang_required)

    # для кроссязыковой векторизации
    if args.lang == 'cross':
        check_args(args, 'method', cross_method_required)

        if args.method == 'translation':
            args.src_embeddings_path = args.tar_embeddings_path

        directions = {d: lang for d, lang in zip(['src', 'tar'], args.direction.split('-'))}
        print(directions)

        print('Векторизую src')
        src_vectorized = main_onelang('src', directions['src'], texts_mapping,
                                      args.src_lemmatized_path, args.src_embeddings_path,
                                      args.src_output_vectors_path, args.method, args.no_duplicates,
                                      args.projection_path, args.bidict_path, args.forced)
        # print(src_vectorized)
        print('Векторизую tar')
        tar_vectorized = main_onelang('tar', directions['tar'], texts_mapping,
                                      args.tar_lemmatized_path, args.tar_embeddings_path,
                                      args.tar_output_vectors_path, args.method, args.no_duplicates,
                                      args.projection_path, args.bidict_path, args.forced)
        # print(tar_vectorized)

        # собираем общие матрицу и маппинг
        common_len = len(src_vectorized) + len(tar_vectorized)
        emb_dim = src_vectorized.shape[1]
        common_vectorized = np.zeros((common_len, emb_dim))
        print(common_vectorized.shape)

        common2i = {}
        i2common = {}

        common_vectorized, common2i, i2common = to_common(texts_mapping, common2i, i2common,
                                                          common_vectorized, tar_vectorized,
                                                          directions['tar'], start_from=0)
        common_vectorized, common2i, i2common = to_common(texts_mapping, common2i, i2common,
                                                          common_vectorized, src_vectorized,
                                                          directions['src'],
                                                          start_from=len(tar_vectorized))

        pdump(common_vectorized, open(args.common_output_vectors_path, 'wb'))

        texts_mapping['cross2i'] = common2i
        texts_mapping['i2cross'] = i2common
        jdump(texts_mapping, open(args.mapping_path, 'w', encoding='utf-8'))

        # print(i2common)
        # print(common2i)

    # для векторизации одноязычного корпуса (без сборки общей матрицы векторов и общего маппинга)
    else:

        check_args(args, 'method', mono_method_required)

        if args.method == 'translation':
            args.src_embeddings_path = args.tar_embeddings_path

        print('Векторизую корпус')
        src_vectorized = main_onelang('src', args.lang, texts_mapping,
                                      args.src_lemmatized_path,
                                      args.src_embeddings_path, args.src_output_vectors_path,
                                      args.method, args.no_duplicates, args.projection_path,
                                      args.bidict_path, args.forced)


if __name__ == "__main__":
    main()
