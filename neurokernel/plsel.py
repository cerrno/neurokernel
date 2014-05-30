#!/usr/bin/env python

"""
Path-like row selector for pandas DataFrames with hierarchical MultiIndexes.
"""

import itertools
import re

import msgpack
import numpy as np
import pandas as pd
import ply.lex as lex
import ply.yacc as yacc

class PathLikeSelector(object):
    """Class for selecting rows of a pandas DataFrame using path-like selectors.

    Select rows from a pandas DataFrame using path-like selectors.
    Assumes that the DataFrame instance has a MultiIndex where each level
    corresponds to a level of the selector. An index level may either be a
    denoted by a string label (e.g., 'foo') or a numerical index (e.g., 0, 1,
    2); a selector level may additionally be a list of strings (e.g.,
    '[foo,bar]') or integers (e.g., '[0,2,4]') or continuous intervals 
    (e.g., '[0:5]'). The '*' symbol matches any value in a level, while a 
    range with an open upper bound (e.g., '[5:]') will match all integers
    greater than or equal to the lower bound.
        
    Examples of valid selectors include

    /foo/bar
    /foo+/bar          (equivalent to /foo/bar)
    /foo/[qux,bar]
    /foo/bar[0]
    /foo/bar/[0]       (equivalent to /foo/bar[0])
    /foo/bar/0         (equivalent to /foo/bar[0])
    /foo/bar[0,1]
    /foo/bar[0:5]
    /foo/*/baz
    /foo/*/baz[5]
    /foo/bar,/baz/qux
    (/foo,/bar)+/baz   (equivalent to /foo/baz,/bar/baz)
    /[foo,bar].+/[0:2] (equivalent to /foo[0],/bar[1])

    An empty string is deemed to be a valid selector.

    The class can also be used to create new MultiIndex instances from selectors
    that can be fully expanded into an explicit set of identifiers (and
    therefore contain no ambiguous symbols such as '*' or '[:]').

    Methods
    -------
    are_consecutive(int_list)
        Check whether a list of integers is consecutive.
    are_disjoint(s0, s1, ...)
        Check whether several selectors are disjoint.
    expand(selector)
        Expand an unambiguous selector into a list of identifiers.
    get_index(df, selector, start=None, stop=None, names=[])
        Return MultiIndex corresponding to rows selected by specified selector.
    get_tuples(df, selector, start=None, stop=None)
        Return tuples containing MultiIndex labels selected by specified selector.
    index_to_selector(idx)
        Convert a MultiIndex into an expanded port selector.
    is_ambiguous(selector)
        Check whether a selector cannot be expanded into an explicit list of identifiers.
    is_expandable(selector)
        Check whether a selector can be expanded into multiple identifiers.
    is_identifier(s)
        Check whether a selector or token sequence can identify a single port.
    is_selector(selector)
        Check whether a string or sequence is a valid selector.
    is_in(s, t)
        Check whether all of the identifiers in one selector are comprised by another.
    make_index(selector, names=[])
        Create a MultiIndex from the specified selector.
    max_levels(selector)
        Return maximum number of token levels in selector.
    parse(selector)
        Parse a selector string into individual port identifiers.
    select(df, selector, start=None, stop=None)
        Select rows from DataFrame using a path-like selector.
    to_identifier(tokens)
        Convert a sequence of tokens into a path-like port identifier string.
    tokenize(selector)
        Tokenize a selector string.

    Notes
    -----
    Since there is no need to maintain multiple instances of the lexer/parser
    used to process path-like selectors, they are associated with the class
    rather than class instances; likewise, all of the class' methods are
    classmethods.

    Numerical indices in path-like selectors are assumed to be
    zero-based. Intervals do not include the end element (i.e., like numpy, not
    like pandas).
    """

    tokens = ('ASTERISK', 'COMMA', 'DOTPLUS', 'INTEGER', 'INTEGER_SET',
              'INTERVAL', 'LPAREN', 'PLUS', 'RPAREN', 'STRING', 'STRING_SET')

    @classmethod
    def _parse_interval_str(cls, s):
        """
        Convert string representation of interval to tuple containing numerical
        start and stop values.
        """

        start, stop = s.split(':')
        if start == '':
            start = 0
        else:
            start = int(start)
        if stop == '':
            stop = np.inf
        else:
            stop = int(stop)
        return (start, stop)

    @classmethod
    def t_PLUS(cls, t):
        r'\+'
        return t

    @classmethod
    def t_DOTPLUS(cls, t):
        r'\.\+'
        return t

    @classmethod
    def t_COMMA(cls, t):
        r'\,'
        return t

    @classmethod
    def t_LPAREN(cls, t):
        r'\('
        return t

    @classmethod
    def t_RPAREN(cls, t):
        r'\)'
        return t

    @classmethod
    def t_ASTERISK(cls, t):
        r'/\*'
        t.value = t.value.strip('/')
        return t

    @classmethod
    def t_INTEGER(cls, t):
        r'/?\d+'
        t.value = int(t.value.strip('/'))
        return t

    @classmethod
    def t_INTEGER_SET(cls, t):
        r'/?\[(?:\d+,?)+\]'
        t.value = map(int, t.value.strip('/[]').split(','))
        return t

    @classmethod
    def t_INTERVAL(cls, t):
        r'/?\[\d*\:\d*\]'
        t.value = cls._parse_interval_str(re.search('\[(.+)\]', t.value).group(1))
        return t

    @classmethod
    def t_STRING(cls, t):
        r'/[^*/\[\]\(\):,\.\d][^+*/\[\]\(\):,\.]*'
        t.value = t.value.strip('/')
        return t

    @classmethod
    def t_STRING_SET(cls, t):
        r'/?\[(?:[^+*/\[\]\(\):,\.\d][^+*/\[\]\(\):,\.]*,?)+\]'
        t.value = t.value.strip('/[]').split(',')
        return t

    @classmethod
    def t_error(cls, t):
        raise ValueError('Cannot tokenize selector - illegal character: %s' % t.value[0])

    # A selector is a list of lists of levels:
    @classmethod
    def p_selector_paren_selector(cls, p):
        'selector : LPAREN selector RPAREN'
        p[0] = p[2]

    @classmethod
    def p_selector_comma_selector(cls, p):
        'selector : selector COMMA selector'
        p[0] = p[1]+p[3]

    @classmethod
    def p_selector_plus_selector(cls, p):
        'selector : selector PLUS selector'
        p[0] = [a+b for a, b in itertools.product(p[1], p[3])]

    @classmethod
    def p_selector_dotplus_selector(cls, p):
        'selector : selector DOTPLUS selector'
        # Expand ranges and wrap strings with lists in each selector:
        for i in xrange(len(p[1])): 
            for j in xrange(len(p[1][i])): 
                if type(p[1][i][j]) in [int, str, unicode]:
                    p[1][i][j] = [p[1][i][j]]
                elif type(p[1][i][j]) == tuple:
                    p[1][i][j] = range(p[1][i][j][0], p[1][i][j][1])
        for i in xrange(len(p[3])):
            for j in xrange(len(p[3][i])):
                if type(p[3][i][j]) in [int, str, unicode]:
                    p[3][i][j] = [p[3][i][j]]
                if type(p[3][i][j]) == tuple:
                    p[3][i][j] = range(p[3][i][j][0], p[3][i][j][1])
                    
        # Fully expand both selectors into individual identifiers
        ids_1 = [list(x) for y in p[1] for x in itertools.product(*y)]
        ids_3 = [list(x) for y in p[3] for x in itertools.product(*y)]
        
        # The expanded selectors must comprise the same number of identifiers:
        assert len(ids_1) == len(ids_3)        
        p[0] = [a+b for (a, b) in zip(ids_1, ids_3)]

    @classmethod
    def p_selector_selector_plus_level(cls, p):
        'selector : selector PLUS level'
        p[0] = [x+[p[3]] for x in p[1]]

    @classmethod
    def p_selector_selector_level(cls, p):
        'selector : selector level'
        p[0] = [x+[p[2]] for x in p[1]]

    @classmethod
    def p_selector_level(cls, p):
        'selector : level'
        p[0] = [[p[1]]]

    @classmethod
    def p_level(cls, p):
        '''level : ASTERISK
                 | INTEGER
                 | INTEGER_SET
                 | INTERVAL
                 | STRING
                 | STRING_SET'''
        p[0] = p[1]

    @classmethod
    def p_error(cls, p):
        raise ValueError('Cannot parse selector - syntax error: %s' % p)

    @classmethod
    def tokenize(cls, selector):
        """
        Tokenize a selector string.

        Parameters
        ----------
        selector : str
            Selector string.

        Returns
        -------
        token_list : list
            List of tokens extracted by ply.
        """

        cls.lexer.input(selector)
        token_list = []
        while True:
            token = cls.lexer.token()
            if not token: break
            token_list.append(token)
        return token_list

    @classmethod
    def parse(cls, selector):
        """
        Parse a selector string into individual port identifiers.

        Parameters
        ----------
        selector : str
            Selector string.

        Returns
        -------
        parse_list : list
            List of lists containing the tokens corresponding to each individual
            selector in the string.
        """

        if re.search('^\s*$', selector):
            return [[]]
        else:
            return cls.parser.parse(selector, lexer=cls.lexer)

    @classmethod
    def is_identifier(cls, s):
        """
        Check whether a selector or token sequence can identify a single port.

        Parameters
        ----------
        s : sequence, str, or unicode
            Selector string (e.g., '/foo[0:2]'), sequence of token sequences
            (e.g., [['foo', (0, 2)]]), or sequence of tokens (e.g., ['foo', 0]).
        
        Returns
        -------
        result : bool
            True for a sequence containing only strings and/or integers
            (e.g., ['foo', 0]) or a selector string that expands into a 
            single sequence of strings and/or integers (e.g., [['foo', 0]]).

        Notes
        -----
        Can check sequences of tokens (even though a sequence of tokens is not a
        valid selector).
        """
        
        if np.iterable(s):
            
            # Try to expand string:
            if type(s) in [str, unicode]:
                try:
                    s_exp = cls.expand(s)
                except:
                    return False
                else:
                    if len(s_exp) == 1:
                        return True
                    else:
                        return False

            # If all entries are iterable non-strings, try to expand:
            elif all([(np.iterable(x) and type(x) not in [str, unicode]) for x in s]):
                if len(cls.expand(s)) == 1:
                    return True
                else:
                    return False

            # A sequence of integers and/or strings is a valid port identifier:
            elif set(map(type, s)).issubset([int, str, unicode]):               
                return True
            else:
                return False

        # A non-iterable cannot be a valid identifier:
        else:
            return False

    @classmethod
    def to_identifier(cls, s):
        """
        Convert an expanded selector or token sequence into a single port identifier string.

        Parameters
        ----------
        s : sequence
            Expanded selector (i.e., a sequence of sequences) or a sequence of 
            string or integer tokens.

        Returns
        -------
        s : str
            Port identifier string.

        Notes
        -----
        Accepts sequences of tokens as well as expanded selectors (even though 
        a sequence of tokens is not a valid selector).
        """

        assert np.iterable(s) and type(s) not in [str, unicode]
        if set(map(type, s)).issubset([int, str, unicode]):
            tokens = s
        else:
            assert len(s) == 1
            tokens = s[0]

        result = ''
        for t in tokens:
            if type(t) == str:
                result += '/'+t
            elif type(t) == int:
                result += '[%s]' % t
            else:
                raise ValueError('Cannot convert to single port identifier.')
        return result

    @classmethod
    def is_ambiguous(cls, selector):
        """
        Check whether a selector cannot be expanded into an explicit list of identifiers.

        A selector is ambiguous if it contains the symbols '*' or '[0:]' (i.e., a
        range with no upper bound).

        Parameters
        ----------
        selector : str or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : bool
            True if the selector is ambiguous, False otherwise.
        """

        if type(selector) in [str, unicode]:
            if re.search(r'(?:\*)|(?:\:\])', selector):
                return True
            else:
                return False
        elif np.iterable(selector):
            for tokens in selector:
                for token in tokens:
                    if token == '*' or \
                       (type(token) == tuple and token[1] == np.inf):
                        return True
            return False
        else:
            raise ValueError('invalid selector type')

    @classmethod
    def is_selector_empty(cls, s):
        """
        Check whether a string or sequence is an empty selector.

        Parameters
        ----------
        s : str, unicode, or sequence
            String or sequence to test.
        
        Returns
        -------
        result : bool
            True if `s` is a sequence containing empty sequences or a null
            string, False otherwise.
        """
        
        if type(s) in [str, unicode]:
            if re.search('^\s*$', s):
                return True
            else:
                return False
        elif np.iterable(s):
            if all(map(np.iterable, s)) and \
               all(map(lambda e: len(e) == 0, s)):
                return True
            else:
                return False
        else:
            return False

    @classmethod
    def is_selector_seq(cls, s):
        """
        Check whether a sequence is a valid selector.

        Parameters
        ----------
        s : sequence
            Sequence to test.

        Returns
        -------
        result : bool
            True if a sequence of valid token sequences
            (e.g., [['foo', (0, 2)]], [['bar', 'baz'], ['qux', 0]]),
            False otherwise.

        Note
        ----
        An empty sequence (e.g., []) is deemed to be a valid selector.
        """

        assert np.iterable(s)
        for tokens in s:

            # The selector must contain sequences of tokens:
            if not np.iterable(tokens):
                return False

            # Each token must either be a string, integer, 2-element tuple,
            # list of strings, or list of integers:
            for token in tokens:
                if type(token) == tuple:
                    if len(token) != 2:
                        return False
                elif type(token) == list:
                    token_types = set(map(type, token))
                    if not (token_types.issubset([str, unicode]) or \
                            token_types == set([int])):
                        return False
                elif type(token) not in [str, unicode, int]:
                    return False

        # All tokens are valid:
        return True
        
    @classmethod
    def is_selector_str(cls, s):
        """
        Check whether a string is a valid selector.

        Parameters
        ----------
        s : str, unicode
            String to test.

        Returns
        -------
        result : bool
            True if the specified selector is a parseable string 
            (e.g., '/foo[0:2]'), False otherwise.
        """

        assert type(s) in [str, unicode]
        try:
            cls.parse(s)
        except:
            return False
        else:
            return True

    @classmethod
    def is_selector(cls, s):
        """
        Check whether a string or sequence is a valid selector.

        Parameters
        ----------
        s : str, unicode, or sequence
            String or sequence to test.

        Returns
        -------
        result : bool
            True if the specified selector is a parseable string (e.g.,
            '/foo[0:2]') or a sequence of valid token sequences.
            (e.g., [['foo', (0, 2)]], [['bar', 'baz'], ['qux', 0]]).
        """

        if type(s) in [str, unicode]:
            return cls.is_selector_str(s)
        elif np.iterable(s):
            return cls.is_selector_seq(s)
        else:
            return False

    @classmethod
    def expand(cls, selector):
        """
        Expand an unambiguous selector into a list of identifiers.

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : list
            List of identifiers; each identifier is a tuple of unambiguous tokens.
        """
        
        assert cls.is_selector(selector)
        assert not cls.is_ambiguous(selector)

        if type(selector) in [str, unicode]:
            p = cls.parse(selector)
        elif np.iterable(selector):
            p = selector
        else:
            raise ValueError('invalid selector type')
        for i in xrange(len(p)):

            # p[i] needs to be mutable:
            p[i] = list(p[i])

            for j in xrange(len(p[i])):
                if type(p[i][j]) in [int, str, unicode]:
                    p[i][j] = [p[i][j]]
                elif type(p[i][j]) == tuple:
                    p[i][j] = range(p[i][j][0], p[i][j][1])
        return [tuple(x) for y in p for x in itertools.product(*y)]

    @classmethod
    def is_expandable(cls, selector):
        """
        Check whether a selector can be expanded into multiple identifiers.

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : bool
            True if the selector contains any intervals or sets of
            strings/integers, False otherwise. Ambiguous selectors are
            not deemed to be expandable, nor are fully expanded selectors.
        """

        assert cls.is_selector(selector)

        if cls.is_ambiguous(selector):
            return False
        if type(selector) in [str, unicode]:
            p = cls.parse(selector)
        elif np.iterable(selector):
            p = selector
        else:
            raise ValueError('invalid selector type')
        for i in xrange(len(p)):
            for j in xrange(len(p[i])):
                if type(p[i][j]) in [int, str, unicode]:
                    p[i][j] = [p[i][j]]

                # Tuples must only contain 2 elements:
                elif type(p[i][j]) == tuple and len(p[i][j]) == 2:
                    p[i][j] = range(p[i][j][0], p[i][j][1])

                    # The presence of a range containing more than 1 element
                    # implies expandability:
                    if len(p[i][j]) > 1: return True                        
                elif type(p[i][j]) == list:

                    # The presence of a list containing more than 1 unique
                    # element implies expandability:
                    if len(set(p[i][j])) > 1: return True
                else:
                    raise ValueError('invalid selector contents')

        if len(set([tuple(x) for y in p for x in itertools.product(*y)])) > 1:
            return True
        else:
            return False
        
    @staticmethod
    def are_consecutive(int_list):
        """
        Check whether a list of integers is consecutive.

        Parameters
        ----------
        int_list : list of int
            List of integers

        Returns
        -------
        result : bool
            True if the integers are consecutive, false otherwise.
        
        Notes
        -----
        Does not assume that the list is sorted.
        """

        if set(np.diff(int_list)) == set([1]):
            return True
        else:
            return False

    @classmethod
    def collapse(cls, id_list):
        """
        Collapse a list of identifiers into a selector string.

        Parameters
        ----------
        id_list : list of tuple
            List of identifiers; each identifier is a list of token tuples.

        Returns
        -------
        selector : str
            String that expands into the given identifier list.

        Notes
        -----
        Expects all identifiers in the given list to have the same
        number of levels.
        """

        # XXX doesn't collapse expanded selectors such as /foo/xxx,/bar/yyy properly

        # Can only collapse list identifiers that all have the same number of
        # levels:
        assert len(set(map(len, id_list))) == 1

        # Collect all tokens for each level:
        levels = [[] for i in xrange(max(map(len, id_list)))]
        for i in xrange(len(id_list)):
            for j in xrange(len(id_list[i])):
                if not(id_list[i][j] in levels[j]):
                    levels[j].append(id_list[i][j])

        def collapse_level(level):
            """
            Recursively called function to collapse all values in a single level.
            """

            type_set = set(map(type, level))
            if type_set == set([int]):

                # If a level only contains consecutive integers, convert it into an
                # interval:
                level.sort()
                if cls.are_consecutive(level):
                    return ['[%s:%s]' % (min(level), max(level)+1)]

                # If a level contains nonconsecutive integers, convert it into a
                # list:
                else:
                    return ['['+','.join([str(i) for i in level])+']']
            elif type_set in set([str, unicode]):
                if len(level) == 1:
                    return level
                else:
                    return ['['+','.join([s for s in level])+']']
            else:
                level_int = sorted([x for x in level if type(x) == int])
                level_str = sorted([x for x in level if type(x) in [str, unicode]])
                return collapse_level(level_int)+collapse_level(level_str)

        # If a level contains multiple string AND integer tokens, convert it to
        # a list:

        collapsed_list = []
        for level in levels:
            collapsed_list.append(collapse_level(sorted(level)))
        selector_list = []
        for t in itertools.product(*collapsed_list):
            selector = ''
            for s in t:
                if s[0] == '[':
                    selector += s
                else:
                    selector = selector + '/' + s
            selector_list.append(selector)
        return ','.join(selector_list)

    @classmethod
    def are_disjoint(cls, *selectors):
        """
        Check whether several selectors are disjoint.

        Parameters
        ----------
        s0, s1, ... : str, unicode, or sequence
            Selectors to check. Each selector is either a string (e.g., 
            '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : bool
            True if none of the identifiers comprised by one selector are
            comprised by the other.

        Notes
        -----
        The selectors must not be ambiguous.
        """

        assert len(selectors) >= 1
        assert all(map(cls.is_selector, selectors))
        if len(selectors) == 1: return True            
        assert all(map(lambda s: not cls.is_ambiguous(s), selectors))

        # Expand selectors into sets of identifiers:
        ids = set()
        for selector in selectors:

            # If some identifiers are present in both the previous expanded
            # selectors and the current selector, the selectors cannot be disjoint:
            ids_new = set(map(tuple, cls.expand(selector)))
            if ids.intersection(ids_new):
                return False
            else:
                ids = ids.union(ids_new)
        return True

    # Need to create cache here because one can't assign create a cache that is
    # an attribute of the classmethod itself:
    __max_levels_cache = {}
    @classmethod
    def max_levels(cls, selector):
        """
        Return maximum number of token levels in selector.

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        count : int
            Maximum number of tokens in selector.
        """

        assert cls.is_selector(selector)

        # Handle unhashable selectors:
        try:
            hash(selector)
        except:
            h = msgpack.dumps(selector)
        else:
            h = selector

        # Use memoization:
        try:
            return cls.__max_levels_cache[h]
        except:
            if type(selector) in [str, unicode]:
                try:
                    count = max(map(len, cls.parse(selector)))
                except:
                    count = 0
            elif np.iterable(selector):
                try:
                    count = max(map(len, selector))
                except:
                    count = 0
            else:
                raise ValueError('invalid selector type')
            cls.__max_levels_cache[h] = count
            return count

    @classmethod
    def _multiindex_row_in(cls, row, parse_list, start=None, stop=None):
        """
        Check whether a row in a MultiIndex matches a parsed selector.

        Check whether the entries in a (subinterval of a) given tuple of data
        corresponding to the entries of one row in a MultiIndex match the
        specified token values.

        Parameters
        ----------
        row : sequence
            Data corresponding to a single row of a MultiIndex.
        parse_list : list
            List of lists of token values extracted by ply.
        start, stop : int
            Start and end indices in `row` over which to test entries. If
            the 

        Returns
        -------
        result : bool
            True of all entries in specified subinterval of row match, 
            False otherwise.
        """

        row_sub = row[start:stop]
        for tokens in parse_list:

            # A single row will never match an empty token list:
            if not tokens:
                continue

            # If this loop terminates prematurely, it will not return True;
            # this forces checking of the subsequent token list:
            for i, token in enumerate(tokens):
                if token == '*':
                    continue
                elif type(token) in [int, str, unicode]:
                    if row_sub[i] != token:
                        break
                elif type(token) == list:
                    if row_sub[i] not in token:
                        break
                elif type(token) == tuple:
                    i_start, i_stop = token
                    if not(row_sub[i] >= i_start and row_sub[i] < i_stop):
                        break
                else:
                    continue
            else:
                return True

        # If the function still hasn't returned, no match was found:
        return False

    @classmethod
    def _index_row_in(cls, row, parse_list):
        """
        Check whether a row in an Index matches a parsed selector.

        Check whether a row label in a MultiIndex matches the
        specified token values.

        Parameters
        ----------
        row : scalar
            Data corresponding to a single row of an Index.
        parse_list : list
            List of lists of token values extracted by ply.

        Returns
        -------
        result : bool
            True of all entries in specified subinterval of row match, 
            False otherwise.
        """

        for tokens in parse_list:
            if not tokens:
                continue
            if len(tokens) > 1:
                raise ValueError('index row only is scalar')
            if tokens[0] == '*':
                continue
            elif type(tokens[0]) in [int, str, unicode]:
                if row != token:
                    break
            elif type(tokens[0]) == list:
                if row not in tokens[0]:
                    break
            elif type(tokens[0]) == tuple:
                i_start, i_stop = tokens[0]
                if not(row >= i_start and row < i_stop):
                    break
            else:
                continue
        else:
            return True

    @classmethod
    def is_in(cls, s, t):
        """
        Check whether all of the identifiers in one selector are comprised by another.
        
        Parameters
        ----------
        s, t : str, unicode, or sequence
            Check whether selector `s` is in `t`. Each selector is either a
            string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : bool
            True if the first selector is in the second, False otherwise.
        """

        assert cls.is_selector(s)
        assert cls.is_selector(t)

        s_exp = set(cls.expand(s))
        t_exp = set(cls.expand(t))
        if s_exp.issubset(t_exp):
            return True
        else:
            return False

    @classmethod
    def get_tuples(cls, df, selector, start=None, stop=None):
        """
        Return tuples containing index labels selected by specified selector.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame instance on which to apply the selector.
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).
        start, stop : int
            Start and end indices in `row` over which to test entries.
            If the index of `df` is an Index, these are ignored.

        Returns
        -------
        result : list
            List of tuples containing index labels for selected rows. If 
            `df.index` is an Index, the result is a list of labels.
        """

        assert cls.is_selector(selector)

        if type(selector) in [str, unicode]:
            parse_list = cls.parse(selector)
        elif np.iterable(selector):
            parse_list = selector
        else:
            raise ValueError('invalid selector type')
        max_levels = max(map(len, parse_list))

        # The maximum number of tokens must not exceed the number of levels in the
        # DataFrame's MultiIndex:        
        if max_levels > len(df.index.names[start:stop]):
            raise ValueError('Maximum number of levels in selector exceeds that of '
                             'DataFrame index')

        if isinstance(df.index, pd.MultiIndex):
            return [t for t in df.index \
                    if cls._multiindex_row_in(t, parse_list, start, stop)]
        else:
            return [t for t in df.index \
                    if cls._index_row_in(t, parse_list)]

    @classmethod
    def get_index(cls, df, selector, start=None, stop=None, names=[]):
        """
        Return index corresponding to rows selected by specified selector.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame instance on which to apply the selector.
        selector : str or unicode
            Row selector.
        start, stop : int
            Start and end indices in `row` over which to test entries.
        names : scalar or list
            Name or names of levels to use in generated index.

        Returns
        -------
        result : pandas.Index or pandas.MultiIndex
            Index that refers to the rows selected by the specified
            selector.
        """

        assert cls.is_selector(selector)

        tuples = cls.get_tuples(df, selector, start, stop)
        if not tuples:
            raise ValueError('no tuples matching selector found')

        # XXX This probably could be made faster by directly manipulating the
        # existing MultiIndex:
        if all(map(np.iterable, tuples)):
            if np.iterable(names) and names:
                return pd.MultiIndex.from_tuples(tuples, names=names)
            elif names:
                return pd.MultiIndex.from_tuples(tuples, names=[names])
            else:
                return pd.MultiIndex.from_tuples(tuples)
        else:
            if np.iterable(names) and names:
                return pd.Index(tuples, name=names[0])
            elif names:
                return pd.Index(tuples, name=names)
            else:
                return pd.Index(tuples)

    @classmethod
    def index_to_selector(cls, idx):
        """
        Convert a MultiIndex into an expanded port selector.

        Parameters
        ----------
        idx : pandas.MultiIndex
            MultiIndex containing port identifiers.
        
        Returns
        -------
        selector : list of tuple
            List of tuples corresponding to individual port identifiers.        
        """

        return idx.tolist()

    @classmethod
    def make_index(cls, selector, names=[]):
        """
        Create a pandas index from the specified selector.

        Parameters
        ----------
        selector : str or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token 
            sequences (e.g., [['foo', (0, 2)]]).            
        names : list
            Names of levels to use in generated MultiIndex. If no names are
            specified, the levels are assigned increasing integers starting with
            0 as their names.

        Returns
        -------
        result : pandas.Index or pandas.MultiIndex
            MultiIndex corresponding to the specified selector. If the selector
            only contains a single level, an Index is returned (this is due to a
            pecularity of pandas).

        Notes
        -----
        The selector may not contain ambiguous symbols such as '*' or 
        '[:]'. 
        """

        assert cls.is_selector(selector)
        assert not cls.is_ambiguous(selector)

        if type(selector) in [str, unicode]:
            parse_list = cls.parse(selector)

        # An empty sequence is not a valid selector:
        elif np.iterable(selector) and selector:
            parse_list = selector
        else:
            raise ValueError('invalid selector type')

        idx_list = []
        for tokens in parse_list:

            # Accumulate lists of tokens for each level:
            list_list = []
            for token in tokens:
                if type(token) == tuple:
                    list_list.append(range(token[0], token[1]))
                elif type(token) == list:
                    list_list.append(token)
                else:
                    list_list.append([token])
            if not names:
                idx_names = range(len(list_list))
            else:
                idx_names = names
            if list_list:
                idx = pd.MultiIndex.from_product(list_list, names=idx_names)
            else:
                idx = pd.Index([])
            idx_list.append(idx)

        # Attempting to create a MultiIndex with a single level results in
        # an Index; therefore, all created indices must either be Index
        # instances or all be MultiIndex instances:        
        if all(map(lambda idx: isinstance(idx, pd.MultiIndex), idx_list)):

            # All of the token lists in the selector must have the same number of
            # levels:
            assert len(set(map(lambda idx: len(idx.levels), idx_list))) == 1
        
            # Combine all of the created indices into a single index:
            return reduce(pd.MultiIndex.append, idx_list)
        if all(map(lambda idx: isinstance(idx, pd.Index) and \
                   not isinstance(idx, pd.MultiIndex), idx_list)):
            return reduce(pd.Index.append, idx_list)
        else:
            raise ValueError('All identifiers must contain same number of levels.')

    @classmethod
    def select(cls, df, selector, start=None, stop=None):
        """
        Select rows from DataFrame using a path-like selector.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame instance on which to apply the selector.
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).            
        start, stop : int
            Start and end indices in `row` over which to test entries.

        Returns
        -------
        result : pandas.DataFrame
            DataFrame containing selected rows.
        """

        assert cls.is_selector(selector)

        if type(selector) in [str, unicode]:
            parse_list = cls.parse(selector)
        elif np.iterable(selector):
            parse_list = selector
        else:
            raise ValueError('invalid selector type')

        # The number of tokens must not exceed the number of levels in the
        # DataFrame's MultiIndex:        
        if len(parse_list) > len(df.index.names[start:stop]):
            raise ValueError('Number of levels in selector exceeds number in row subinterval')

        if type(df.index) == pd.MultiIndex:
            return df.select(lambda row: cls._multiindex_row_in(row, parse_list, 
                                                                start, stop))
        else:
            return df.select(lambda row: cls._index_row_in(row, parse_list))

# Set the option optimize=1 in the production version; need to perform these
# assignments after definition of the rest of the class because the class'
# internal namespace can't be accessed within its body definition:
PathLikeSelector.lexer = lex.lex(module=PathLikeSelector)
PathLikeSelector.parser = yacc.yacc(module=PathLikeSelector, 
                                    debug=0, write_tables=0)

class PortMapper(object):
    """
    Maps a numpy array to/from path-like port identifiers.

    Parameters
    ----------
    data : numpy.ndarray
        Data to map to ports.
    selector : str, unicode, or sequence
        Selector string (e.g., '/foo[0:2]') or sequence of token sequences
        (e.g., [['foo', (0, 2)]]) to map to `data`.
    idx : sequence
        Indices of elements in the specified array to map to ports. If no
        indices are specified, the entire array is mapped to the ports specified
        by the given selector.

    Notes
    -----
    The selectors may not contain any '*' or '[:]' characters.
    """

    def __init__(self, data, selector, idx=None):

        # Can currently only handle unidimensional data structures:
        assert np.ndim(data) == 1
        assert type(data) == np.ndarray

        # Save a reference to the specified array:
        self.data = data

        self.sel = PathLikeSelector()
        if idx is None:
            self.portmap = pd.Series(data=np.arange(len(data)))
        else:
            self.portmap = pd.Series(data=np.asarray(idx))        
        self.portmap.index = self.sel.make_index(selector)

    def get(self, selector):
        """
        Retrieve mapped data specified by given selector.

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).

        Returns
        -------
        result : numpy.ndarray
            Selected data.
        """

        return self.data[self.sel.select(self.portmap, selector).values]

    def set(self, selector, data):
        """
        Set mapped data specified by given selector.

        Parameters
        ----------
        selector : str, unicode, or sequence
            Selector string (e.g., '/foo[0:2]') or sequence of token sequences
            (e.g., [['foo', (0, 2)]]).            
        data : numpy.ndarray
            Array of data to save.
        """

        self.data[self.sel.select(self.portmap, selector).values] = data

    __getitem__ = get
    __setitem__ = set

    def __repr__(self):
        return 'map:\n'+self.portmap.__repr__()+'\n\ndata:\n'+self.data.__repr__()

if __name__ == '__main__':
    from unittest import main, TestCase
    from pandas.util.testing import assert_frame_equal, assert_index_equal

    df1 = pd.DataFrame(data={'data': np.random.rand(12),
                       'level_0': ['foo', 'foo', 'foo', 'foo', 'foo', 'foo',
                                   'bar', 'bar', 'bar', 'bar', 'baz', 'baz'],
                       'level_1': ['qux', 'qux', 'qux', 'qux', 'mof', 'mof',
                                   'qux', 'qux', 'qux', 'mof', 'mof', 'mof'],
                       'level_2': ['xxx', 'yyy', 'yyy', 'yyy', 'zzz', 'zzz',
                                   'xxx', 'xxx', 'yyy', 'zzz', 'yyy', 'zzz'],
                       'level_3': [0, 0, 1, 2, 0, 1,
                                   0, 1, 0, 1, 0, 1]})
    df1.set_index('level_0', append=False, inplace=True)
    df1.set_index('level_1', append=True, inplace=True)
    df1.set_index('level_2', append=True, inplace=True)
    df1.set_index('level_3', append=True, inplace=True)

    df = pd.DataFrame(data={'data': np.random.rand(10),
                      0: ['foo', 'foo', 'foo', 'foo', 'foo',
                          'bar', 'bar', 'bar', 'baz', 'baz'],
                      1: ['qux', 'qux', 'mof', 'mof', 'mof',
                          'qux', 'qux', 'qux', 'qux', 'mof'],
                      2: [0, 1, 0, 1, 2, 
                          0, 1, 2, 0, 0]})
    df.set_index(0, append=False, inplace=True)
    df.set_index(1, append=True, inplace=True)
    df.set_index(2, append=True, inplace=True)

    class test_path_like_selector(TestCase):
        def setUp(self):
            self.df = df.copy()
            self.sel = PathLikeSelector()

        def test_select_str(self):
            result = self.sel.select(self.df, '/foo')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_list(self):
            result = self.sel.select(self.df, [['foo']])
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_comma(self):
            result = self.sel.select(self.df, '/foo/qux,/baz/mof')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('baz','mof',0)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_plus(self):
            result = self.sel.select(self.df, '/foo+/qux+[0,1]')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_dotplus(self):
            result = self.sel.select(self.df, '/[bar,baz].+/[qux,mof].+/[0,0]')
            idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                             ('baz','mof',0)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_paren(self):
            result = self.sel.select(self.df, '(/bar,/baz)')
            idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                             ('bar','qux',1),
                                             ('bar','qux',2),
                                             ('baz','qux',0),
                                             ('baz','mof',0)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_paren_plus(self):
            result = self.sel.select(self.df, '(/bar,/baz)+/qux')
            idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                             ('bar','qux',1),
                                             ('bar','qux',2),
                                             ('baz','qux',0)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_asterisk(self):
            result = self.sel.select(self.df, '/*/qux')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('bar','qux',0),
                                             ('bar','qux',1),
                                             ('bar','qux',2),
                                             ('baz','qux',0)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_integer_with_brackets(self):
            result = self.sel.select(self.df, '/bar/qux[1]')
            idx = pd.MultiIndex.from_tuples([('bar','qux',1)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_integer_no_brackets(self):
            result = self.sel.select(self.df, '/bar/qux/1')
            idx = pd.MultiIndex.from_tuples([('bar','qux',1)], names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_integer_set(self):
            result = self.sel.select(self.df, '/foo/qux[0,1]')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_string_set(self):
            result = self.sel.select(self.df, '/foo/[qux,mof]')
            idx = pd.MultiIndex.from_tuples([('foo','qux',0),
                                             ('foo','qux',1),
                                             ('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_interval_no_bounds(self):
            result = self.sel.select(self.df, '/foo/mof[:]')
            idx = pd.MultiIndex.from_tuples([('foo','mof',0),
                                             ('foo','mof',1),
                                             ('foo','mof',2)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_interval_lower_bound(self):
            result = self.sel.select(self.df, '/foo/mof[1:]')
            idx = pd.MultiIndex.from_tuples([('foo','mof',1),
                                             ('foo','mof',2)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_interval_upper_bound(self):
            result = self.sel.select(self.df, '/foo/mof[:2]')
            idx = pd.MultiIndex.from_tuples([('foo','mof',0),
                                             ('foo','mof',1)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_select_interval_both_bounds(self):
            result = self.sel.select(self.df, '/bar/qux[0:2]')
            idx = pd.MultiIndex.from_tuples([('bar','qux',0),
                                             ('bar','qux',1)],
                                            names=[0, 1, 2])
            assert_frame_equal(result, self.df.ix[idx])

        def test_are_disjoint_str(self):
            assert self.sel.are_disjoint('/foo[0:10]/baz',
                                         '/bar[10:20]/qux') == True
            assert self.sel.are_disjoint('/foo[0:10]/baz',
                                         '/foo[5:15]/[baz,qux]') == False

        def test_are_disjoint_list(self):
            result = self.sel.are_disjoint([['foo', (0, 10), 'baz']], 
                                           [['bar', (10, 20), 'qux']])
            assert result == True
            result = self.sel.are_disjoint([['foo', (0, 10), 'baz']], 
                                           [['foo', (5, 15), ['baz','qux']]])
            assert result == False

        def test_expand_str(self):
            result = self.sel.expand('/foo/bar[0:2],/moo/[qux,baz]')
            self.assertSequenceEqual(result,
                                     [('foo', 'bar', 0),
                                      ('foo', 'bar', 1),
                                      ('moo', 'qux'), 
                                      ('moo', 'baz')])

        def test_expand_list(self):
            result = self.sel.expand([['foo', 'bar', (0, 2)],
                                      ['moo', ['qux', 'baz']]])
            self.assertSequenceEqual(result,
                                     [('foo', 'bar', 0),
                                      ('foo', 'bar', 1),
                                      ('moo', 'qux'), 
                                      ('moo', 'baz')])

        def test_get_index_str(self):
            idx = self.sel.get_index(self.df, '/foo/mof/*')
            assert_index_equal(idx, pd.MultiIndex(levels=[['foo'], ['mof'],
                                                          [0, 1, 2]],
                                                  labels=[[0, 0, 0],
                                                          [0, 0, 0],
                                                          [0, 1, 2]]))

        def test_get_index_list(self):
            idx = self.sel.get_index(self.df, [['foo', 'mof', '*']])
            assert_index_equal(idx, pd.MultiIndex(levels=[['foo'], ['mof'],
                                                          [0, 1, 2]],
                                                  labels=[[0, 0, 0],
                                                          [0, 0, 0],
                                                          [0, 1, 2]]))

        def test_get_tuples_str(self):
            result = self.sel.get_tuples(df, '/foo/mof/*')
            self.assertSequenceEqual(result,
                                     [('foo', 'mof', 0),
                                      ('foo', 'mof', 1),
                                      ('foo', 'mof', 2)])

        def test_get_tuples_list(self):
            result = self.sel.get_tuples(df, [['foo', 'mof', '*']])
            self.assertSequenceEqual(result,
                                     [('foo', 'mof', 0),
                                      ('foo', 'mof', 1),
                                      ('foo', 'mof', 2)])
            
        def test_is_ambiguous_str(self):
            assert self.sel.is_ambiguous('/foo/*') == True
            assert self.sel.is_ambiguous('/foo/[5:]') == True
            assert self.sel.is_ambiguous('/foo/[:10]') == False
            assert self.sel.is_ambiguous('/foo/[5:10]') == False

        def test_is_ambiguous_list(self):
            assert self.sel.is_ambiguous([['foo', '*']]) == True
            assert self.sel.is_ambiguous([['foo', (5, np.inf)]]) == True
            assert self.sel.is_ambiguous([['foo', (0, 10)]]) == False
            assert self.sel.is_ambiguous([['foo', (5, 10)]]) == False

        def test_is_identifier(self):
            assert self.sel.is_identifier('/foo/bar') == True
            assert self.sel.is_identifier(0) == False
            assert self.sel.is_identifier('foo') == False
            #assert self.sel.is_identifier('0') == False # this doesn't work
            assert self.sel.is_identifier(['foo', 'bar']) == True
            assert self.sel.is_identifier(['foo', 0]) == True
            assert self.sel.is_identifier(['foo', [0, 1]]) == False
            assert self.sel.is_identifier([['foo', 'bar']]) == True
            assert self.sel.is_identifier([['foo', 'bar'], ['baz']]) == False
            assert self.sel.is_identifier([['foo', 0]]) == True

        def test_to_identifier(self):
            assert self.sel.to_identifier(['foo']) == '/foo'
            assert self.sel.to_identifier(['foo', 0]) == '/foo[0]'
            self.assertRaises(Exception, self.sel.to_identifier, 'foo')
            self.assertRaises(Exception, self.sel.to_identifier, 
                              [['foo', ['a', 'b']]])
            self.assertRaises(Exception, self.sel.to_identifier, 
                              ['foo', (0, 2)])

        def test_is_in_str(self):
            assert self.sel.is_in('/foo/bar[5]', '/[foo,baz]/bar[0:10]') == True
            assert self.sel.is_in('/qux/bar[5]', '/[foo,baz]/bar[0:10]') == False

        def test_is_in_list(self):
            assert self.sel.is_in([['foo', 'bar', [5]]], 
                                   [[['foo', 'baz'], 'bar', (0, 10)]]) == True
            assert self.sel.is_in([['qux', 'bar', [5]]], 
                                   [[['foo', 'baz'], 'bar', (0, 10)]]) == False

        def test_is_selector_empty(self):
            assert self.sel.is_selector_empty('') == True            
            assert self.sel.is_selector_empty([[]]) == True
            assert self.sel.is_selector_empty([()]) == True
            assert self.sel.is_selector_empty([[], []]) == True
            assert self.sel.is_selector_empty([(), []]) == True

            assert self.sel.is_selector_empty('/foo') == False
            assert self.sel.is_selector_empty([['foo', 'bar']]) == False
            assert self.sel.is_selector_empty([['']]) == False # is this correct?

        def test_is_selector_str(self):
            assert self.sel.is_selector('') == True
            assert self.sel.is_selector('/foo') == True
            assert self.sel.is_selector('/foo/bar') == True
            assert self.sel.is_selector('/foo!?') == True
            assert self.sel.is_selector('/foo[0]') == True
            assert self.sel.is_selector('/foo[0:2]') == True
            assert self.sel.is_selector('/foo[0:]') == True
            assert self.sel.is_selector('/foo[:2]') == True
            assert self.sel.is_selector('/foo/*') == True
            assert self.sel.is_selector('/foo,/bar') == True
            assert self.sel.is_selector('/foo+/bar') == True
            assert self.sel.is_selector('/foo[0:2].+/bar[0:2]') == True

            assert self.sel.is_selector('/foo[') == False
            assert self.sel.is_selector('foo[0]') == False

        def test_is_selector_list(self):
            assert self.sel.is_selector([[]]) == True
            assert self.sel.is_selector([['foo', 'bar']]) == True
            assert self.sel.is_selector([('foo', 'bar')]) == True
            assert self.sel.is_selector([('foo', '*')]) == True
            assert self.sel.is_selector([('foo', 'bar'), ('bar', 'qux')]) == True
            assert self.sel.is_selector([('foo', 0)]) == True
            assert self.sel.is_selector([('foo', (0, 2))]) == True
            assert self.sel.is_selector([('foo', (0, np.inf))]) == True
            assert self.sel.is_selector([('foo', [0, 1])]) == True
            assert self.sel.is_selector([('foo', ['a', 'b'])]) == True

            assert self.sel.is_selector([('foo', (0, 1, 2))]) == False
            assert self.sel.is_selector([('foo', 'bar'),
                                         ((0, 1, 2), 0)]) == False
            assert self.sel.is_selector([('foo', ['a', 0])]) == False

        def test_make_index_str_single_level(self):
            idx = self.sel.make_index('/foo')
            assert_index_equal(idx, pd.Index(['foo'], dtype='object'))
            idx = self.sel.make_index('/foo,/bar')
            assert_index_equal(idx, pd.Index(['foo', 'bar'], dtype='object'))

        def test_make_index_str_multiple_levels(self):
            idx = self.sel.make_index('/[foo,bar]/[0:3]')
            assert_index_equal(idx, pd.MultiIndex(levels=[['bar', 'foo'],
                                                          [0, 1, 2]],
                                                  labels=[[1, 1, 1, 0, 0, 0],
                                                          [0, 1, 2, 0, 1, 2]]))

        def test_make_index_list_single_level(self):
            idx = self.sel.make_index([['foo']])
            assert_index_equal(idx, pd.Index(['foo'], dtype='object'))
            idx = self.sel.make_index([['foo'], ['bar']])
            assert_index_equal(idx, pd.Index(['foo', 'bar'], dtype='object'))

        def test_make_index_list_multiple_levels(self):
            idx = self.sel.make_index([[['foo', 'bar'], (0, 3)]])
            assert_index_equal(idx, pd.MultiIndex(levels=[['bar', 'foo'],
                                                          [0, 1, 2]],
                                                  labels=[[1, 1, 1, 0, 0, 0],
                                                          [0, 1, 2, 0, 1, 2]]))

        def test_make_index_invalid(self):
            self.assertRaises(Exception, self.sel.make_index, 'foo/bar[')
            self.assertRaises(Exception, self.sel.make_index, 
                              [['foo', 'bar'], ['baz']])
            self.assertRaises(Exception, self.sel.make_index, 
                              [['foo', 'bar', (0, 2)], ['baz', 'qux']])

        def test_max_levels_str(self):
            assert self.sel.max_levels('/foo/bar[0:10]') == 3
            assert self.sel.max_levels('/foo/bar[0:10],/baz/qux') == 3

        def test_max_levels_list(self):
            assert self.sel.max_levels([['foo', 'bar', (0, 10)]]) == 3
            assert self.sel.max_levels([['foo', 'bar', (0, 10)],
                                        ['baz', 'qux']]) == 3

    class test_port_mapper(TestCase):
        def setUp(self):
            self.data = np.random.rand(20)

        def test_get(self):
            pm = PortMapper(self.data,
                            '/foo/bar[0:10],/foo/baz[0:10]')
            np.allclose(self.data[0:10], pm['/foo/bar[0:10]'])

        def test_get_discontinuous(self):
            pm = PortMapper(self.data,
                            '/foo/bar[0:10],/foo/baz[0:10]')
            np.allclose(self.data[[0, 2, 4, 6]],
                        pm['/foo/bar[0,2,4,6]'])

        def test_get_sub(self):
            pm = PortMapper(self.data,
                            '/foo/bar[0:5],/foo/baz[0:5]',
                            np.arange(5, 15))
            np.allclose(self.data[5:10], pm['/foo/bar[0:5]'])

        def test_set(self):
            pm = PortMapper(self.data,
                            '/foo/bar[0:10],/foo/baz[0:10]')
            pm['/foo/baz[0:5]'] = 1.0
            np.allclose(np.ones(5), pm['/foo/baz[0:5]'])

        def test_set_discontinuous(self):
            pm = PortMapper(self.data,
                            '/foo/bar[0:10],/foo/baz[0:10]')
            pm['/foo/*[0:2]'] = 1.0
            np.allclose(np.ones(4), pm['/foo/*[0:2]'])

    main()
