"""
Работа с аргументами от argparse
"""

from utils.errors import RequiredArgumentLost


def check_args(args, key_arg, required_dict):
    """
    Проверка, не потерялись ли необходимые аргументы (требования в required_dicts)
    :param args: аргумнты от argparse
    :param key_arg: аргумент, значения которого влекут за собой разные требования
    :param required_dict: список требований
    :return: выбрасывает ошибку, если какой-то аргумент потерян, и указывает, что должно быть
    """
    key_value = args.__getattribute__(key_arg)
    if key_value in required_dict:
        required_args = required_dict[key_value]
        # собираем обязательные аргументы, которых нет
        lost_args = {arg for arg in required_args if not args.__getattribute__(arg)}
        if lost_args:
            raise RequiredArgumentLost(key_arg, key_value, required_args, lost_args)


def arg_to_list(arg, splitter = ','):
    """
    Превращение строки от argparse в список чисел
    """
    if isinstance(arg, list):
        return arg
    else:
        new_arg = [int(x) for x in arg.split(splitter)]
        return new_arg
