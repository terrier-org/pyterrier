from enum import Enum
from typing import List, Union
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

class BaseTokeniser():
    LOWERCASE = True
    maxNumOfSameConseqLettersPerTerm = 3
    maxNumOfDigitsPerTerm = 4
    DROP_LONG_TOKENS = True
    MAX_TERM_LENGTH = 20

class UTFTokeniser(BaseTokeniser):
    @staticmethod
    def check(s : str) -> str:
        # if the s is None
        # or if it is longer than a specified length
        s = s.strip()
        length = len(s)
        counter = 0
        counterdigit = 0
        ch = -1
        chNew = -1
        for i in range(length):
            chNew = s[i]
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
    def tokenise(input : str) -> List[str]:
        ch : int
        index : int = -1
        def read() -> Union[int, str]:
            nonlocal index
            if index == len(input) -1:
                return -1
            index += 1
            return input[index]
        
        ch = read()
        eos : bool = False
        def stream_tokenise():
            nonlocal eos, ch
            while ch != -1:
                # skip non-alphanumeric or space characters
                while ch != -1 and not (ch.isalnum() or unicodedata.category(ch) in ['Mn', 'Mc']): # "Mark, Nonspacing", "Mark, Spacing Combining"
                    ch = read()

                    
                sw = []  # Using list to build string
                # now accept all alphanumeric characters
                while ch != -1 and (ch.isalnum() or unicodedata.category(ch) in ['Mn', 'Mc']):
                    # add character to word so far
                    sw.append(ch)
                    ch = read()
                   
                if len(sw) > BaseTokeniser.MAX_TERM_LENGTH:
                    if BaseTokeniser.DROP_LONG_TOKENS:
                        return None
                    else:
                        sw = sw[:BaseTokeniser.MAX_TERM_LENGTH]

                s = UTFTokeniser.check(''.join(sw))
                if len(s) > 0:
                    return s
            eos = True
            return None
            
        rtr = []
        while not eos:
            token = stream_tokenise()
            if token is not None:
                rtr.append(token)
        return rtr


class EnglishTokeniser(BaseTokeniser):
    

    @staticmethod
    def check(s : str) -> str:
        # if the s is None or if it is longer than a specified length
        s = s.strip()
        length = len(s)
        counter = 0
        counterdigit = 0
        ch = -1
        chNew = -1
        for i in range(length):
            chNew = ord(s[i])
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
    def tokenise(input : str) -> List[str]:
        ch : int
        index : int = -1
        def read() -> Union[int,str]:
            nonlocal index
            if index == len(input) -1:
                return -1
            index += 1
            return input[index]
        ch = read()
        eos : bool = False
        def stream_tokenise():
            nonlocal eos, ch
            while ch != -1:
                # skip non-alphanumeric characters
                while ch != -1 and not (('A' <= ch <= 'Z') or ('a' <= ch<= 'z') or ('0' <= ch <= '9')):
                    ch = read()

                sw = []  # Using a list to build the string
                # now accept all alphanumeric characters
                while ch != -1 and (('A' <= ch <= 'Z') or ('a' <= ch <= 'z') or ('0' <= ch <= '9')):
                    # add character to word so far
                    sw.append(ch)
                    ch = read()

                if len(sw) > BaseTokeniser.MAX_TERM_LENGTH:
                    if BaseTokeniser.DROP_LONG_TOKENS:
                        return None
                    else:
                        sw = sw[:BaseTokeniser.MAX_TERM_LENGTH]  # Truncate the list to MAX_TERM_LENGTH

                s = EnglishTokeniser.check(''.join(sw))
                if len(s) > 0:
                    return s

            eos = True
            return None
        rtr = []
        while not eos:
            token = stream_tokenise()
            if token is not None:
                rtr.append(token)
        return rtr