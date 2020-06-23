class UnknownMethodError(Exception):
    """
    Указан неизвестный метод векторизации
    """
    def __init__(self):
        self.text = 'Неправильно указан метод векторизации! ' \
                    'Варианты: "model", "projection", "translation"'

    def __str__(self):
        return self.text


class NotLemmatizedError(Exception):
    """
    По указанному пути не нашёлся json с лемматизированными тестами
    """
    def __init__(self):
        self.text = 'Нечего векторизовать! Пожалуйста, сначала лемматизируйте тексты'

    def __str__(self):
        return self.text


class NotIncludedError(Exception):
    """
    Указали included=True для текста, которого нет в маппинге
    """
    def __init__(self):
        self.text = 'Такого текста нет в корпусе! ' \
                    'Пожалуйста, измените значение параметра included'

    def __str__(self):
        return self.text


class NoModelProvided(Exception):
    """
    Забыли указать путь к модели лемматизации или векторизации
    """
    def __init__(self):
        self.text = 'Пожалуйста, укажите пути ' \
                    'к моделям для лемматизации и векторизации текста.'

    def __str__(self):
        return self.text


class RequiredArgumentLost(Exception):
    """
    Забыли указать параметр, обязательный при других параметрах
    """
    def __init__(self, key_arg, key_value, required_args, lost_args):
        self.text = "Параметр {} = {} требует следующих параметров: {}\nПожалуйста, укажите {}"\
            .format(key_arg, key_value, ', '.join(required_args), ', '.join(lost_args))

    def __str__(self):
        return self.text
