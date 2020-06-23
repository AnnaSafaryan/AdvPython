"""
Классы для векторизации различными методами (моделью, проекцией, переводом)
Функция для определения, какой власс вызвать
"""

import numpy as np
from numpy.linalg import norm

from utils.loaders import load_embeddings, load_projection, load_bidict
from utils.errors import UnknownMethodError


class BaseVectorizer:
    def __init__(self, embeddings_path, no_duplicates):
        self.embeddings_file = embeddings_path
        self.no_duplicates = no_duplicates
        self.model = load_embeddings(embeddings_path)
        self.dim = self.model.vector_size
        self.empty = np.zeros(self.dim)  # TODO: возвращать вектор какого-то слова

    def __str__(self):
        return '{}:\n\tModel: {}\n\tDim: {}\n\tDuplicates: {}'.format(self.__class__.__name__, self.embeddings_file, self.dim, not self.no_duplicates)

    def get_words(self, tokens):
        words = [token for token in tokens if token in self.model]
        # если прилетел пустой текст, то он так и останется пустым просто

        if self.no_duplicates:
            words = set(words)

        return words

    def get_norm_vec(self, vec):
        vec = vec / norm(vec)
        return vec

    def get_mean_vec(self, vecs):
        vec = np.sum(vecs, axis=0)
        vec = np.divide(vec, len(vecs))
        return vec

    # def get_norm_mean_vec(self, vecs):
    #     vec = self.get_mean_vec(vecs)
    #     vec = self.get_norm_vec(vec)
    #     return vec

    def __floordiv__(self, vecs):
        '''нормируем средний вектор через //'''
        vec = self.get_mean_vec(vecs)
        vec = self.get_norm_vec(vec)
        return vec

    def base_vectorize_text(self, tokens):
        # простая векторизация моделью
        words = self.get_words(tokens)

        if not words:
            print('Я ничего не знаю из этих токенов: {}'.format(tokens))
            self.__setattr__('', False)
            return self.empty

        t_vecs = np.zeros((len(words), self.dim))
        for i, token in enumerate(words):
            t_vecs[i, :] = self.model[token]
        t_vec = self//t_vecs

        return t_vec

    def __lshift__(self, tokens):
        return self.vectorize_text(tokens)


class ModelVectorizer(BaseVectorizer):
    '''векторизация текста моделью'''

    def __init__(self, embeddings_path, no_duplicates):
        super(ModelVectorizer, self).__init__(embeddings_path, no_duplicates)

    def vectorize_text(self, tokens):
        t_vec = self.base_vectorize_text(tokens)
        return t_vec


class ProjectionVectorizer(BaseVectorizer):
    '''векторизация текста матрицей трансформации'''

    def __init__(self, embeddings_path, projection_path, no_duplicates):
        super(ProjectionVectorizer, self).__init__(embeddings_path, no_duplicates)
        self.projection = load_projection(projection_path)
    
    def project_vec(self, src_vec):
        '''Проецируем вектор'''

        test = np.mat(src_vec)
        test = np.c_[1.0, test]  # Adding bias term
        predicted_vec = np.dot(self.projection, test.T)
        predicted_vec = np.squeeze(np.asarray(predicted_vec))
        # print('Проецирую и нормализую вектор')
        return predicted_vec

    def predict_projection_word(self, src_word, tar_emdedding, topn=10):
        '''По слову предсказываем переводы и трансформированный вектор'''
        src_vec = self.model[src_word]
        predicted_vec = self.project_vec(src_vec)
        # нашли ближайшие в другой модели
        nearest_neighbors = tar_emdedding.most_similar(positive=[predicted_vec], topn=topn)
        return nearest_neighbors, predicted_vec

    def vectorize_text(self, tokens):
        '''векторизация текста матрицей трансформации'''

        words = self.get_words(tokens)

        if not words:
            print('Я ничего не знаю из этих токенов: {}'.format(tokens))
            return self.empty

        t_vecs = np.zeros((len(words), self.dim))
        for i, token in enumerate(words):
            src_vec = self.model[token]
            t_vecs[i, :] = self.project_vec(src_vec)
        t_vec = self//t_vecs

        return t_vec


class TranslationVectorizer(BaseVectorizer):
    '''Перевод русского текста на английский и обычная векторизация'''

    def __init__(self, embeddings_path, bidict_path, no_duplicates):
        super(TranslationVectorizer, self).__init__(embeddings_path, no_duplicates)
        self.bidict = load_bidict(bidict_path)
        # print(self.bidict)

    def translate_text(self, text):
        '''Переводим лемматизировнный русский корпус по лемматизированному двуязычному словарю'''
        # print(text)
        translated_text = [self.bidict[word] for word in text if word in self.bidict]
        # print(translated_text)
        return translated_text

    def vectorize_text(self, tokens):
        '''переведённый текст векторизуем базовой функцией'''
        translated_tokens = self.translate_text(tokens)
        t_vec = self.base_vectorize_text(translated_tokens)
        return t_vec


def build_vectorizer(direction, method, embeddings_path, no_duplicates,
                     projection_path='', bidict_path=''):
    if direction == 'tar':  # для целевого языка всегда векторизуем моделью
        vectorizer = ModelVectorizer(embeddings_path, no_duplicates)

    else:  # для исходного языка уже определяем метод
        if method == 'model':  # если векторизуем предобученной моделью
            vectorizer = ModelVectorizer(embeddings_path, no_duplicates)
        elif method == 'projection':
            vectorizer = ProjectionVectorizer(embeddings_path, projection_path, no_duplicates)
        elif method == 'translation':
            vectorizer = TranslationVectorizer(embeddings_path, bidict_path, no_duplicates)
        else:
            raise UnknownMethodError()

    print(vectorizer)
    return vectorizer

