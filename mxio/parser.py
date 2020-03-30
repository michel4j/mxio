import os
import re
from collections import defaultdict

import yaml

SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ESCAPE_CHARS = ")("


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


class Int(Converter):
    name = 'int'
    regex = '[0-9]+'

    @staticmethod
    def to_python(value):
        return int(value.strip())


class String(Converter):
    name = 'str'
    regex = '.+'

    @staticmethod
    def to_python(value):
        return value.strip()


class Slug(Converter):
    name = 'slug'
    regex = '[-a-zA-Z0-9_]+'

    @staticmethod
    def to_python(value):
        return value.strip()


class Float(Converter):
    name = 'float'
    regex = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

    @staticmethod
    def to_python(value):
        return float(value.strip())


def escape(text):
    for c in ESCAPE_CHARS:
        text = text.replace(c, '\{}'.format(c))
    return text


def build(pattern):
    """
    Parse the text and generate the corresponding regex expression, replacing all fields
    :param pattern: parser specification
    @return:
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
        if 'size' in variable:
            regex = r'\s*?(?P<{}>.{{{}}})\s*?'.format(variable['key'], variable['size'])
        else:
            regex = r'\s*?(?P<{}>{})\s*?'.format(variable['key'], variable.get('regex', variable['converter'].regex))
        pattern = pattern.replace(variable['token'], regex, 1)
    return re.compile(pattern), variables


def parse_fields(spec, text):
    groups = defaultdict(list)
    regex, variables = build(spec)
    converters = {variable['key']: variable['converter'] for variable in variables}
    for variable in variables:
        groups[variable['name']].append(variable['key'])

    m = regex.search(text)
    if m:
        raw_values = {k: converters[k].to_python(v) for k, v in m.groupdict().items()}
        return {
            name: raw_values[name] if len(keys) == 1 else tuple(raw_values[key] for key in keys)
            for name, keys in groups.items()
        }
    return {}


def parse_section(section, data):
    data_patt = re.compile(r'({}.+?{})'.format(section.get('start', '^'), section.get('end', '$')), re.DOTALL)
    m = data_patt.search(data)
    output = {}
    if m:
        sub_data = m.group(0)
        if 'fields' in section:
            if isinstance(section['fields'], list):
                for spec in section['fields']:
                    output.update(parse_fields(spec, sub_data))
            elif isinstance(section['fields'], dict):
                for sub_name, sub_section in section['fields'].items():
                    output[sub_name] = parse_section(sub_section, sub_data)
    return output


def parse_text(data, spec_name):
    spec_file = '{}.yml'.format(spec_name)
    with open(os.path.join(SPEC_PATH, spec_file), 'r') as handle:
        specs = yaml.safe_load(handle)
    return parse_section(specs['root'], data)


def parse(data_file, spec_name, size=-1):
    with open(data_file, 'r') as handle:
        data = handle.read(size)
    return parse_text(data, spec_name)
