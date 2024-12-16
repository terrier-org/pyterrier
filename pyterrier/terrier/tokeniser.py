import re
from enum import Enum
from typing import List

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
    MAX_TERM_LENGTH = 20

# _MN_MC are all characters that belong to the "Mark, Nonspacing", "Mark, Spacing Combining" unicode categories,
# as a regular expression. It was generated with the following code:
# import unicodedata
# marks = [c for c in range(0x110000) if unicodedata.category(chr(c)) in {'Mn', 'Mc'}]
# ranges = []
# rng = [0,0]
# for c in marks:
#   if c == rng[1] + 1:
#     rng[1] = c
#   else:
#     rng = [c, c]
#     ranges.append(rng)
# def utfenc(c):
#   c = format(c, '04x')
#   if len(c) > 4:
#     return r'\U' + c.zfill(8)
#   return r'\u' + c
# regex = ''
# for x, y in ranges:
#   regex += utfenc(x)
#   if x != y:
#     regex += '-' + utfenc(y)
_MN_MC = r'\u0300-\u036f\u0483-\u0487\u0591-\u05bd\u05bf\u05c1-\u05c2\u05c4-\u05c5\u05c7\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7-\u06e8\u06ea-\u06ed\u0711\u0730-\u074a\u07a6-\u07b0\u07eb-\u07f3\u07fd\u0816-\u0819\u081b-\u0823\u0825-\u0827\u0829-\u082d\u0859-\u085b\u0898-\u089f\u08ca-\u08e1\u08e3-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\u0981-\u0983\u09bc\u09be-\u09c4\u09c7-\u09c8\u09cb-\u09cd\u09d7\u09e2-\u09e3\u09fe\u0a01-\u0a03\u0a3c\u0a3e-\u0a42\u0a47-\u0a48\u0a4b-\u0a4d\u0a51\u0a70-\u0a71\u0a75\u0a81-\u0a83\u0abc\u0abe-\u0ac5\u0ac7-\u0ac9\u0acb-\u0acd\u0ae2-\u0ae3\u0afa-\u0aff\u0b01-\u0b03\u0b3c\u0b3e-\u0b44\u0b47-\u0b48\u0b4b-\u0b4d\u0b55-\u0b57\u0b62-\u0b63\u0b82\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcd\u0bd7\u0c00-\u0c04\u0c3c\u0c3e-\u0c44\u0c46-\u0c48\u0c4a-\u0c4d\u0c55-\u0c56\u0c62-\u0c63\u0c81-\u0c83\u0cbc\u0cbe-\u0cc4\u0cc6-\u0cc8\u0cca-\u0ccd\u0cd5-\u0cd6\u0ce2-\u0ce3\u0cf3\u0d00-\u0d03\u0d3b-\u0d3c\u0d3e-\u0d44\u0d46-\u0d48\u0d4a-\u0d4d\u0d57\u0d62-\u0d63\u0d81-\u0d83\u0dca\u0dcf-\u0dd4\u0dd6\u0dd8-\u0ddf\u0df2-\u0df3\u0e31\u0e34-\u0e3a\u0e47-\u0e4e\u0eb1\u0eb4-\u0ebc\u0ec8-\u0ece\u0f18-\u0f19\u0f35\u0f37\u0f39\u0f3e-\u0f3f\u0f71-\u0f84\u0f86-\u0f87\u0f8d-\u0f97\u0f99-\u0fbc\u0fc6\u102b-\u103e\u1056-\u1059\u105e-\u1060\u1062-\u1064\u1067-\u106d\u1071-\u1074\u1082-\u108d\u108f\u109a-\u109d\u135d-\u135f\u1712-\u1715\u1732-\u1734\u1752-\u1753\u1772-\u1773\u17b4-\u17d3\u17dd\u180b-\u180d\u180f\u1885-\u1886\u18a9\u1920-\u192b\u1930-\u193b\u1a17-\u1a1b\u1a55-\u1a5e\u1a60-\u1a7c\u1a7f\u1ab0-\u1abd\u1abf-\u1ace\u1b00-\u1b04\u1b34-\u1b44\u1b6b-\u1b73\u1b80-\u1b82\u1ba1-\u1bad\u1be6-\u1bf3\u1c24-\u1c37\u1cd0-\u1cd2\u1cd4-\u1ce8\u1ced\u1cf4\u1cf7-\u1cf9\u1dc0-\u1dff\u20d0-\u20dc\u20e1\u20e5-\u20f0\u2cef-\u2cf1\u2d7f\u2de0-\u2dff\u302a-\u302f\u3099-\u309a\ua66f\ua674-\ua67d\ua69e-\ua69f\ua6f0-\ua6f1\ua802\ua806\ua80b\ua823-\ua827\ua82c\ua880-\ua881\ua8b4-\ua8c5\ua8e0-\ua8f1\ua8ff\ua926-\ua92d\ua947-\ua953\ua980-\ua983\ua9b3-\ua9c0\ua9e5\uaa29-\uaa36\uaa43\uaa4c-\uaa4d\uaa7b-\uaa7d\uaab0\uaab2-\uaab4\uaab7-\uaab8\uaabe-\uaabf\uaac1\uaaeb-\uaaef\uaaf5-\uaaf6\uabe3-\uabea\uabec-\uabed\ufb1e\ufe00-\ufe0f\ufe20-\ufe2f\U000101fd\U000102e0\U00010376-\U0001037a\U00010a01-\U00010a03\U00010a05-\U00010a06\U00010a0c-\U00010a0f\U00010a38-\U00010a3a\U00010a3f\U00010ae5-\U00010ae6\U00010d24-\U00010d27\U00010eab-\U00010eac\U00010efd-\U00010eff\U00010f46-\U00010f50\U00010f82-\U00010f85\U00011000-\U00011002\U00011038-\U00011046\U00011070\U00011073-\U00011074\U0001107f-\U00011082\U000110b0-\U000110ba\U000110c2\U00011100-\U00011102\U00011127-\U00011134\U00011145-\U00011146\U00011173\U00011180-\U00011182\U000111b3-\U000111c0\U000111c9-\U000111cc\U000111ce-\U000111cf\U0001122c-\U00011237\U0001123e\U00011241\U000112df-\U000112ea\U00011300-\U00011303\U0001133b-\U0001133c\U0001133e-\U00011344\U00011347-\U00011348\U0001134b-\U0001134d\U00011357\U00011362-\U00011363\U00011366-\U0001136c\U00011370-\U00011374\U00011435-\U00011446\U0001145e\U000114b0-\U000114c3\U000115af-\U000115b5\U000115b8-\U000115c0\U000115dc-\U000115dd\U00011630-\U00011640\U000116ab-\U000116b7\U0001171d-\U0001172b\U0001182c-\U0001183a\U00011930-\U00011935\U00011937-\U00011938\U0001193b-\U0001193e\U00011940\U00011942-\U00011943\U000119d1-\U000119d7\U000119da-\U000119e0\U000119e4\U00011a01-\U00011a0a\U00011a33-\U00011a39\U00011a3b-\U00011a3e\U00011a47\U00011a51-\U00011a5b\U00011a8a-\U00011a99\U00011c2f-\U00011c36\U00011c38-\U00011c3f\U00011c92-\U00011ca7\U00011ca9-\U00011cb6\U00011d31-\U00011d36\U00011d3a\U00011d3c-\U00011d3d\U00011d3f-\U00011d45\U00011d47\U00011d8a-\U00011d8e\U00011d90-\U00011d91\U00011d93-\U00011d97\U00011ef3-\U00011ef6\U00011f00-\U00011f01\U00011f03\U00011f34-\U00011f3a\U00011f3e-\U00011f42\U00013440\U00013447-\U00013455\U00016af0-\U00016af4\U00016b30-\U00016b36\U00016f4f\U00016f51-\U00016f87\U00016f8f-\U00016f92\U00016fe4\U00016ff0-\U00016ff1\U0001bc9d-\U0001bc9e\U0001cf00-\U0001cf2d\U0001cf30-\U0001cf46\U0001d165-\U0001d169\U0001d16d-\U0001d172\U0001d17b-\U0001d182\U0001d185-\U0001d18b\U0001d1aa-\U0001d1ad\U0001d242-\U0001d244\U0001da00-\U0001da36\U0001da3b-\U0001da6c\U0001da75\U0001da84\U0001da9b-\U0001da9f\U0001daa1-\U0001daaf\U0001e000-\U0001e006\U0001e008-\U0001e018\U0001e01b-\U0001e021\U0001e023-\U0001e024\U0001e026-\U0001e02a\U0001e08f\U0001e130-\U0001e136\U0001e2ae\U0001e2ec-\U0001e2ef\U0001e4ec-\U0001e4ef\U0001e8d0-\U0001e8d6\U0001e944-\U0001e94a\U000e0100-\U000e01ef'

class UTFTokeniser(BaseTokeniser):
    """
    Port of the functionality of https://github.com/terrier-org/terrier-core/blob/5.x/modules/core/src/main/java/org/terrier/indexing/tokenisation/UTFTokeniser.java to Python
    """
    # the following finds sequences of that are either alphanumeric or "Mark, Nonspacing", "Mark, Spacing Combining" characters.
    # Note that it also includes underscores, which need to be checked for separately. (But I couldn't find a way to do the check
    # efficiently in the regular expression itself.)
    RE_TOKEN = re.compile(rf'[\w{_MN_MC}]+')

    @staticmethod
    def check(s : str) -> str:
        # if the s is None
        # or if it is longer than a specified length
        s = s.strip()
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
        result = []
        for match in UTFTokeniser.RE_TOKEN.finditer(input):
            for s in match.group().split('_'):
                s = UTFTokeniser.check(s)
                if 0 < len(s) <= BaseTokeniser.MAX_TERM_LENGTH:
                    result.append(s)
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
