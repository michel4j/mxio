import os
import re

from pathlib import Path
from collections import defaultdict
from typing import Union, List, Tuple

import yaml

SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ESCAPE_CHARS = ")(.*|"


class ConverterType(type):
    def __init__(cls, *args, **kwargs):
        super(ConverterType, cls).__init__(*args, **kwargs)
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        if hasattr(cls, 'name'):
            cls.registry[cls.name] = cls

    def __getitem__(cls, item):
        return cls.registry[item]

    def __contains__(cls, item):
        return item in cls.registry

    def get(cls, item):
        return cls.registry[item]

    def get_types(self):
        return self.registry


class Converter(object, metaclass=ConverterType):
    """
    Converter Base Class
    """
    chars = r"."

    @classmethod
    def regex(cls, name, size=None):
        if size is None:
            return rf'\s*?(?P<{name}>{cls.chars}+)\s*?'
        else:
            return rf'{cls.chars}{{1,{size}}}'


class Int(Converter):
    name = 'int'
    chars = r'[\d]'

    @classmethod
    def regex(cls, name, size=None):
        if not size:
            return rf'\s*?(?P<{name}>[-+]?\d+)\s*?'
        else:
            return rf'(?<=\s)(?=.{{{size}}})(?P<{name}>\s*[-+]?\d+)(?=[^\d])'

    @staticmethod
    def to_python(value):
        return int(value.strip())


class String(Converter):
    name = 'str'
    chars = r'.'

    @staticmethod
    def to_python(value):
        return value.strip()


class Char(Converter):
    name = 'char'
    chars = r'.'

    @staticmethod
    def to_python(value):
        return value

    @classmethod
    def regex(cls, name, size=None):
        if not size:
            return rf'(?P<{name}>{cls.chars})'
        else:
            return rf'(?P<{name}>{cls.chars}{{{size}}})'


class Slug(Converter):
    name = 'slug'
    chars = r'[-a-zA-Z0-9_/.]'

    @staticmethod
    def to_python(value):
        return value.strip()


class Float(Converter):
    name = 'float'

    @classmethod
    def regex(cls, name, size=None):
        if not size:
            return rf'\s*?(?P<{name}>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*?'
        else:
            return rf'(?<=\s)(?=.{{{size}}})(?P<{name}>\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?=[^\d])'

    @staticmethod
    def to_python(value):
        return float(value.strip())


class Line(Converter):
    name = 'line'

    @classmethod
    def regex(cls, name, size=None):
        return rf'^|\n(?P<{name}>[^\n]*)\n|$'

    @staticmethod
    def to_python(value):
        return value


def escape(text):
    for c in ESCAPE_CHARS:
        text = text.replace(c, r'\{}'.format(c))
    return text


def build_pattern(pattern: str) -> Tuple[re.Pattern, List[dict]]:
    """
    Parse the text and generate the corresponding regex expression, replacing all fields
    :param pattern: parser specification
    :return: compiled pattern and list of field parameters
    """
    field_pattern = re.compile(r'^(?P<type>\w+):(?P<name>\w+)(?::(?P<regex>\(.*?\)))?(?::(?P<size>\d+))?$')
    tokens = re.findall(r'<([^<>]+?)>', pattern)
    counts = defaultdict(int)
    variables = []
    # extract token parameters
    for token in tokens:
        match = field_pattern.match(token)
        if not match: continue

        # Extract token fields
        data = {k: v for k, v in match.groupdict().items() if v is not None}
        data['token'] = '<{}>'.format(token)

        if data['type'] not in Converter:
            continue
        data['converter'] = Converter[data['type']]
        counts[data['name']] += 1
        variables.append(data)

    tuples = [key for key, count in counts.items() if count > 1]
    index = defaultdict(int)

    # select key name for regex, and prepare pattern
    pattern = escape(pattern)

    for variable in variables:
        if variable['name'] in tuples:
            variable['key'] = '{}_{}'.format(variable['name'], index[variable['name']])
            index[variable['name']] += 1
        else:
            variable['key'] = variable['name']

        # Build field regex and replace token in pattern
        if 'regex' in variable:
            regex = r'(?P<{}>{})'.format(variable['key'], variable['regex'])
        else:
            regex = variable['converter'].regex(variable['key'], variable.get('size'))
        pattern = pattern.replace(variable['token'], regex, 1)
    return re.compile(pattern), variables


def parse_fields(spec: str, text: str, table: bool = False) -> Union[dict, List[dict]]:
    """
    Execute an atomic parser field specification on a text file and return a dictionary of results with proper key-value pairs
    of the right type.
    :param spec: A field specification string. For example: ""CRYSTAL MOSAICITY (DEGREES) <float:mosaicity>""
    :param text: string to parse
    :param table: Whether to parse it as a table
    :return: dictionary of matched key value pairs or list of dictionaries if a table
    """

    groups = defaultdict(list)
    regex, variables = build_pattern(spec)
    converters = {variable['key']: variable['converter'] for variable in variables}
    for variable in variables:
        # ignore internal names
        if not variable['name'].startswith('_'):
            groups[variable['name']].append(variable['key'])

    if table:
        results = []
        for m in regex.finditer(text):
            raw_values = {k: converters[k].to_python(v) for k, v in m.groupdict().items() if not k.startswith('_')}
            results.append({
                name: raw_values[name] if len(keys) == 1 else tuple(raw_values[key] for key in keys)
                for name, keys in groups.items()
            })
        return results
    else:
        m = regex.search(text)
        if m:
            raw_values = {k: converters[k].to_python(v) for k, v in m.groupdict().items() if not k.startswith('_')}
            return {
                name: raw_values[name] if len(keys) == 1 else tuple(raw_values[key] for key in keys)
                for name, keys in groups.items()
            }
    return {}


def parse_text(specs: dict, text: str) -> dict:
    """
    Execute nested parser specification hierarchy on a text file and return the corresponding nested dictionary of
    matched key-value pairs
    :param specs: A nested dictionary of specifications
    :param text: text file
    :return: nested dictionary of key-value pairs
    """

    if specs.get('domains'):
        sub_data = '\n'.join(re.findall(specs["domains"], text, re.DOTALL))
    elif specs.get('domain'):
        m = re.search(specs["domain"], text, re.DOTALL)
        if m:
            sub_data = m.group(0)
        else:
            sub_data = ""
    else:
        sub_data = text

    output = {}
    if sub_data:
        if 'fields' in specs:
            if isinstance(specs['fields'], list):
                for spec in specs['fields']:
                    output.update(parse_fields(spec, sub_data))
        elif 'lines' in specs:
            if isinstance(specs['lines'], list):
                spec = '\\n' + '\\n'.join(specs['lines'])
                output.update(parse_fields(spec, sub_data))
        elif 'table' in specs:
            if isinstance(specs['table'], list):
                spec = '\\n' + '\\n'.join(specs['table'])
            else:
                spec = specs['table']
            return parse_fields(spec, sub_data, table=True)
        if 'sections' in specs:
            for sub_name, sub_section in specs['sections'].items():
                output[sub_name] = parse_text(sub_section, sub_data)
    return output


def parse_file(data_file: Union[str, Path], specs: dict, size: int = -1) -> dict:
    """
    Parse a text file and return the matched dictionary
    :param data_file: file path or name
    :param specs: A nested dictionary of specifications
    :param size: maximum size of file to parse in bytes
    :return: nested dictionary of key-value pairs
    """
    with open(data_file, 'r', encoding='utf-8') as handle:
        text = handle.read(size)
    return parse_text(specs, text)
