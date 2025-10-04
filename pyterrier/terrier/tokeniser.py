import re
from enum import Enum
from typing import List
import unicodedata

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

class BaseTokeniser():
    LOWERCASE = True
    maxNumOfSameConseqLettersPerTerm = 3
    maxNumOfDigitsPerTerm = 4
    MAX_TERM_LENGTH = 20


class UTFTokeniser(BaseTokeniser):
    """
    Port of the functionality of https://github.com/terrier-org/terrier-core/blob/5.x/modules/core/src/main/java/org/terrier/indexing/tokenisation/UTFTokeniser.java to Python
    """
    _TR = None
    _RE = re.compile(r'1+')

    @staticmethod
    def check(s : str) -> str:
        # if the s is None
        # or if it is longer than a specified length
        counter = 0
        counterdigit = 0
        ch = -1
        for chNew in s:
            if chNew.isdigit():
                counterdigit += 1
            if ch == chNew:
                counter += 1
            else:
                counter = 1
            ch = chNew
            # if it contains more than 4 consecutive same letters,
            # or more than 4 digits, then discard the term.
            if counter > BaseTokeniser.maxNumOfSameConseqLettersPerTerm or counterdigit > BaseTokeniser.maxNumOfDigitsPerTerm:
                return ""
        return s.lower() if BaseTokeniser.LOWERCASE else s

    @staticmethod
    def tokenise(input: str) -> List[str]:
        if UTFTokeniser._TR is None:
            # build a translation table that maps the token character classes to "1" and other characters to "0"
            utf_tok_chr_cls_map = {'Lu': '1', 'Ll': '1', 'Lt': '1', 'Lm': '1', 'Lo': '1', 'Mn': '1', 'Mc': '1', 'Nd': '1'}
            max_utf = 0x110000
            UTFTokeniser._TR = [
                (49 if c <= 0xFFFF and unicodedata.category(chr(c)) in utf_tok_chr_cls_map else 48) # 48="0" 49="1"
                for c in range(max_utf)
            ]
        result = []
        for match in UTFTokeniser._RE.finditer(input.translate(UTFTokeniser._TR)):
            s, e = match.span()
            t = input[s:e]
            t = UTFTokeniser.check(t)
            if 0 < len(t) <= BaseTokeniser.MAX_TERM_LENGTH:
                result.append(t)
        return result


class EnglishTokeniser(BaseTokeniser):
    """
    Port of the functionality of https://github.com/terrier-org/terrier-core/blob/5.x/modules/core/src/main/java/org/terrier/indexing/tokenisation/EnglishTokeniser.java to Python
    """
    RE_TOKEN = re.compile(r'[A-Za-z0-9]+')

    @staticmethod
    def check(s : str) -> str:
        # if the s is None or if it is longer than a specified length
        s = s.strip()
        counter = 0
        counterdigit = 0
        ch = -1
        for c in s:
            chNew = ord(c)
            if 48 <= chNew <= 57:  # 0 to 9
                counterdigit += 1
            if ch == chNew:
                counter += 1
            else:
                counter = 1
            ch = chNew
            # if it contains more than 3 consecutive same letters,
            # or more than 4 digits, then discard the term.
            if counter > BaseTokeniser.maxNumOfSameConseqLettersPerTerm or counterdigit > BaseTokeniser.maxNumOfDigitsPerTerm:
                return ""
        return s.lower() if BaseTokeniser.LOWERCASE else s
    
    @staticmethod
    def tokenise(input: str) -> List[str]:
        result = []
        for match in EnglishTokeniser.RE_TOKEN.finditer(input):
            s = EnglishTokeniser.check(match.group())
            if 0 < len(s) <= BaseTokeniser.MAX_TERM_LENGTH:
                result.append(s)
        return result
        raise ValueError(f'Unsupported tokeniser: {this}')
