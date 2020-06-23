'''
Загрузка всякой всячины (эмбеддинги, словари, проекции, лемматизированные корпуса...)
'''

from gensim import models
import logging
import numpy as np
import zipfile
import os
from json import load as jload
from pickle import load as pload
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_embeddings(embeddings_path):
    """
    :param embeddings_path: путь к модели эмбеддингов (строка)
    :return: загруженная предобученная модель эмбеддингов (KeyedVectors)
    """
    # Бинарный формат word2vec:
    if embeddings_path.endswith('.bin.gz') or embeddings_path.endswith('.bin'):
        model = models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True,
                                                         unicode_errors='replace')
    # Текстовый формат word2vec:
    elif embeddings_path.endswith('.txt.gz') or embeddings_path.endswith('.txt') \
            or embeddings_path.endswith('.vec.gz') or embeddings_path.endswith('.vec'):
        model = models.KeyedVectors.load_word2vec_format(
            embeddings_path, binary=False, unicode_errors='replace')

    # ZIP-архив:
    elif embeddings_path.endswith('.zip'):
        with zipfile.ZipFile(embeddings_path, "r") as archive:
            stream = archive.open("model.bin")  # или model.txt, чтобы взглянуть на модель
            model = models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors='replace')
    else:
        model = models.KeyedVectors.load(embeddings_path)

    model.init_sims(replace=True)
    return model


def load_projection(projection_path):
    projection = np.loadtxt(projection_path, delimiter=',')
    return projection


def load_bidict(bidict_path):
    '''читаем словарь пар в словарбь'''
    print('path', bidict_path)
    lines = open(bidict_path, encoding='utf-8').read().splitlines()
    bidict = {line.split()[0]: line.split()[1] for line in lines}
    # print(len(lines))
    return bidict


def load_article_data(article_data_path):
    '''получаем словарь хеш: название, ссылка'''
    lines = open(article_data_path, encoding='utf-8').read().splitlines()
    article_data = {line.split('\t')[0]:
                    {'real_title': line.split('\t')[1], 'url': line.split('\t')[2]} for line in lines}
    # print(article_data)
    return article_data


def load_mapping(mapping_path, forced):
    if os.path.isfile(mapping_path) and not forced:
        mapping = jload(open(mapping_path, 'r', encoding='utf-8'))
        print('Уже есть какой-то маппинг!')
        print('\t'.join(['{}: {} объекта'.format(k, len(v)) for k, v in mapping.items()]))
    else:
        mapping = {}
        print('Маппинга ещё нет, сейчас будет')
    return mapping


def load_lemmatized(lemmatized_path, forced):
    """
    :param lemmatized_path: путь к словарю заголовков и лемм (строка)
    :param forced: принудительно лемматизировать весь корпус заново (pseudo-boolean int)
    :return: словарь заголовков и лемм
    """
    # если существует уже разбор каких-то файлов
    if os.path.isfile(lemmatized_path) and not forced:
        lemmatized = jload(open(lemmatized_path, encoding='utf-8'))
        print('Уже что-то разбирали!')

    else:  # ничего ещё из этого корпуса не разбирали или принудительно обновляем всё
        lemmatized = {}
        print('Ничего ещё не разбирали, сейчас будем')

    return lemmatized


def collect_texts(lemmatized, texts_path, files):
    '''собираем тексты из папки в словарь'''
    for file in tqdm(files):
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').\
            read().lower().strip().split()
        lemmatized[file] = text
    return lemmatized


def get_lemmatized_corpus(mapping, i2lang_name, lemmatized, n_new):
    """собираем корпус лемматизированных текстов и [], если в маппинге есть, а в lemmatized нет"""
    corpus = []
    # для каждого номера в маппинге от последнего в vectorized
    for nr in range(len(mapping[i2lang_name]) - n_new, len(mapping[i2lang_name])):
        # порядок текстов -- как в индексах
        title = mapping[i2lang_name].get(str(nr))
        # по номеру из маппинга берём название и находим его в леммах, если нет -- []
        lemmatized_text = lemmatized.get(title, [])
        corpus.append(lemmatized_text)
    return corpus


def load_vectorized(output_vectors_path, forced):
    """загрузка матрицы с векторами корпуса, если есть"""
    # если существует уже какой-то векторизованный корпус
    if os.path.isfile(output_vectors_path) and not forced:
        vectorized = pload(open(output_vectors_path, 'rb'))
        print('Уже что-то векторизовали!')

    else:  # ничего ещё из этого корпуса не векторизовали или принудительно обновляем всё
        print('Ничего ещё не разбирали, сейчас будем.')
        vectorized = []

    return vectorized


def load_golden_standart(path, mapping, lang2i_name):
    """
    :param path: путь к json с золотым стандартом (строка)
    :param mapping: маппинг заголовков в индексы и обратно (словарь словарей)
    :param lang2i_name: название словаря индекс-заголовок в маппинге (строка)
    :return: индексы заголовков в порядке близости к данному (список)
    """
    raw = (open(path, encoding='utf-8')).read().lower().splitlines()
    titles = {line.split('\t')[0]: line.split('\t')[1] for line in raw}
    ids = {mapping[lang2i_name].get(art): mapping[lang2i_name].get(sim_art)
           for art, sim_art in titles.items()}
    return ids