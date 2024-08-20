from enum import Enum


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
    def _to_obj(this):
        try:
            return TerrierTokeniser(this)
        except ValueError:
            return this

    @staticmethod
    def _to_class(this):
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
