
def class_distribution(data, class_column):
    """
    Возвращает количество объектов, распределенных по классам.

    Параметры:
    - data: DataFrame, содержащий данные.
    - class_column: str, название столбца с классами.

    Возвращает:
    - positive_count: int, количество объектов положительного класса.
    - negative_count: int, количество объектов отрицательного класса.
    """
    positive_count = sum(data[class_column] == 1)
    negative_count = sum(data[class_column] == 0)
    return positive_count, negative_count
