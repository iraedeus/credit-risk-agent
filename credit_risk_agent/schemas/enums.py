"""
Enum definitions for credit risk agent categorical features.
"""

from enum import IntEnum


class Sex(IntEnum):
    """
    Sex categorical feature enumeration.

    Attributes
    ----------
    MALE : int
        Male gender (code 1).
    FEMALE : int
        Female gender (code 2).
    """

    MALE = 1
    FEMALE = 2


class Education(IntEnum):
    """
    Education level categorical feature enumeration.

    Attributes
    ----------
    GRADUATE_SCHOOL : int
        Graduate school education level (code 1).
    UNIVERSITY : int
        University education level (code 2).
    HIGH_SCHOOL : int
        High school education level (code 3).
    OTHERS : int
        Other education levels (code 4).
    """

    GRADUATE_SCHOOL = 1
    UNIVERSITY = 2
    HIGH_SCHOOL = 3
    OTHERS = 4


class Marriage(IntEnum):
    """
    Marital status categorical feature enumeration.

    Attributes
    ----------
    MARRIED : int
        Married status (code 1).
    SINGLE : int
        Single status (code 2).
    OTHERS : int
        Other marital statuses (code 3).
    """

    MARRIED = 1
    SINGLE = 2
    OTHERS = 3
