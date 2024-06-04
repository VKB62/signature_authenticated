from django.test import TestCase
from random import uniform


def r() -> float:
    return uniform(98.2, 99.7)
