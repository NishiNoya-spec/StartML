import math
import random
from typing import Sequence


def example_1():
    """Breakpoint, посмотрим на значения, подсчитаем что-то над ними."""
    a = 5
    print(f'my_value: {a}')
    b = 3
    return a + b


def example_2(a: int):
    """Step over."""
    if a < 4:
        print("into first branch")
        return "foo"
    else:
        print("into second branch")
        print(1)
        return "bar"


def example_2_modified(a: int):
    """Помощник к example_3."""
    example_1()
    return example_2(a)


def example_3(a: int, b: int):
    """Step into."""
    if a > b:
        tmp = example_2(a)
        return f"{b}{tmp}"
    else:
        tmp = example_2(b)
        return f"{tmp}{a}"


def example_4(big_list: Sequence[int]):
    """Go to the line."""
    result = 0
    processed_numbers = []
    for i in big_list:
        if i >= 10:
            result += i**2
        elif 2 <= i < 10:
            result += i**3
        else:
            result += i**4
        processed_numbers.append(i)
    return result


def example_5(a: int):
    """Pause program."""
    while True:
        if a > 5:
            break


def example_6(arr: Sequence[int]):
    """Stacktrace in debugger."""

    def quick_sort(arr: Sequence[int]):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        a_left = [x for x in arr if x < pivot]
        a_mid = [x for x in arr if x == pivot]
        a_right = [x for x in arr if x > pivot]
        left_result = quick_sort(a_left)
        right_result = quick_sort(a_right)
        return left_result + a_mid + right_result

    return quick_sort(arr)


def example_7():
    """pandas, numpy and matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    a = pd.DataFrame(
        [["Morzh", 3, 123321], ["Misha", 14, 300_000]],
        columns=["name", "age", "salary"],
    )
    b = np.array([[5, 10, 3, 4], [4, 123, 59, 3]])
    plt.plot(a["age"], a["salary"])
    print(1)


def inner_f(x):
    x = math.sin(x)
    return x**2 + 5 * x + 5


def example_8():
    """(advanced) breakpoints in a multithread program."""
    # dummy сделает ThreadPool https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.dummy
    from multiprocessing import Pool
    from multiprocessing.dummy import Pool as ThreadPool

    with Pool(3) as p:
        result = p.map(inner_f, [2, 4, 5, 10, 48, -4, 18])
    print(result)

    with ThreadPool(3) as p:
        result = p.map(inner_f, [2, 4, 5, 10, 48, -4, 18])
    print(result)


if __name__ == "__main__":
    # example_1()

    example_2(5)
    # example_2(10)
    #
    # example_3(10, 5)
    # example_3(15, 20)
    #
    # example_4([random.randint(-100, 20) for _ in range(1000)])
    # example_5(1)
    # example_6([4, 1, 3, 2, -1, 4, 10])
    # example_7()
    # example_8()
    print(1)
