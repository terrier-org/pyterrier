import re
from enum import Enum
from typing import List
import unicodedata
import pyterrier as pt

class TerrierTokeniser(Enum):
    """
        This enum provides an API for the tokeniser configuration used during indexing with Terrier.
    """

    whitespace = 'whitespace' #: Tokenise on whitespace only
    english = 'english' #: Terrier's standard tokeniser, designed for English
    utf = 'utf' #: A variant of Terrier's standard tokeniser, similar to English, but with UTF support.
    twitter = 'twitter' #: Like utf, but keeps hashtags etc
    identity = 'identity' #: Performs no tokenisation - strings are kept as is. 

    @staticmethod
    def _to_obj(this) -> 'TerrierTokeniser':
        try:
            return TerrierTokeniser(this)
        except ValueError:
            return this

    @staticmethod
    def _to_class(this) -> str:
        if this == TerrierTokeniser.whitespace:
            return 'WhitespaceTokeniser'
        if this == TerrierTokeniser.english:
            return 'EnglishTokeniser'
        if this == TerrierTokeniser.utf:
            return 'UTFTokeniser'
        if this == TerrierTokeniser.twitter:
            return 'UTFTwitterTokeniser'
        if this == TerrierTokeniser.identity:
            return 'IdentityTokeniser'
        if isinstance(this, str):
            return this
        raise ValueError(f'Unknown/unsupported tokeniser: {this}')
    
    @staticmethod
    @pt.java.required()
    def java_tokeniser(this):
        clz = TerrierTokeniser._to_class(this)
        if "." not in clz:
            clz = "org.terrier.indexing.tokenisation." + clz
        return pt.java.autoclass(clz)()
