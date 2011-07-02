# Based on https://gist.github.com/317164 and PyYAML sources
# (MIT Licence; GPL compatible)

# This module supersedes yaml, extending it to support OrderedDicts as
# the base mapping type
from yaml import *

import yaml
import yaml.representer
import yaml.constructor

import collections

def _construct_odict(load, node):
	"""This is the same as SafeConstructor.construct_yaml_omap(),
	except the data type is changed to collections.OrderedDict() and setitem is
	used instead of append in the loop.

	>>> yaml.load('''
	... !!omap
	... - foo: bar
	... - mumble: quux
	... - baz: gorp
	... ''')
	collections.OrderedDict([('foo', 'bar'), ('mumble', 'quux'), ('baz', 'gorp')])

	>>> yaml.load('''!!omap [ foo: bar, mumble: quux, baz : gorp ]''')
	collections.OrderedDict([('foo', 'bar'), ('mumble', 'quux'), ('baz', 'gorp')])
	"""

	omap = collections.OrderedDict()
	yield omap
	if not isinstance(node, yaml.SequenceNode):
		raise yaml.constructor.ConstructorError(
			"while constructing an ordered map",
			node.start_mark,
			"expected a sequence, but found %s" % node.id, node.start_mark
		)
	for subnode in node.value:
		if not isinstance(subnode, yaml.MappingNode):
			raise yaml.constructor.ConstructorError(
				"while constructing an ordered map", node.start_mark,
				"expected a mapping of length 1, but found %s" % subnode.id,
				subnode.start_mark
			)
		if len(subnode.value) != 1:
			raise yaml.constructor.ConstructorError(
				"while constructing an ordered map", node.start_mark,
				"expected a single mapping item, but found %d items" % len(subnode.value),
				subnode.start_mark
			)
		key_node, value_node = subnode.value[0]
		key = load.construct_object(key_node)
		value = load.construct_object(value_node)
		omap[key] = value

def _construct_mapping(load, node, deep=False):
	"""
	Load regular mappings as OrderedDicts
	"""
	mapping = collections.OrderedDict()
	yield mapping
	if not isinstance(node, yaml.MappingNode):
		raise yaml.constructor.ConstructorError("while constructing a mapping",
			node.start_mark,
			'expected a mapping node, but found %s' % node.id, node.start_mark)

	for key_node, value_node in node.value:
		key = load.construct_object(key_node, deep=deep)
		try:
			hash(key)
		except TypeError, exc:
			raise yaml.constructor.ConstructorError('while constructing a mapping',
				node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
		value = load.construct_object(value_node, deep=deep)
		mapping[key] = value

def _repr_odict(dumper, data):
	"""
	>>> data = collections.OrderedDict([('foo', 'bar'), ('mumble', 'quux'), ('baz', 'gorp')])
	>>> yaml.dump(data, default_flow_style=False)
	'!!omap\\n- foo: bar\\n- mumble: quux\\n- baz: gorp\\n'
	>>> yaml.dump(data, default_flow_style=True)
	'!!omap [foo: bar, mumble: quux, baz: gorp]\\n'
	"""
	return _repr_pairs(dumper, u'tag:yaml.org,2002:omap', data.iteritems())

def _repr_pairs(dump, tag, sequence, flow_style=None):
	"""This is the same code as BaseRepresenter.represent_sequence(),
	but the value passed to dump.represent_data() in the loop is a
	dictionary instead of a tuple."""

	value = []
	node = yaml.SequenceNode(tag, value, flow_style=flow_style)
	if dump.alias_key is not None:
		dump.represented_objects[dump.alias_key] = node
	best_style = True
	for (key, val) in sequence:
		item = dump.represent_data({key: val})
		if not (isinstance(item, yaml.ScalarNode) and not item.style):
			best_style = False
		value.append(item)
	if flow_style is None:
		if dump.default_flow_style is not None:
			node.flow_style = dump.default_flow_style
		else:
			node.flow_style = best_style
	return node

def _repr_dict(dumper, data):
	return represent_mapping(dumper, u'tag:yaml.org,2002:map', data)

def represent_mapping(dump, tag, mapping, flow_style=None):
	value = []
	node = yaml.MappingNode(tag, value, flow_style=flow_style)
	if dump.alias_key is not None:
		dump.represented_objects[dump.alias_key] = node
	best_style = True
	for item_key, item_value in mapping.iteritems():
		node_key = dump.represent_data(item_key)
		node_value = dump.represent_data(item_value)
		if not (isinstance(node_key, yaml.ScalarNode) and not node_key.style):
			best_style = False
		if not (isinstance(node_value, yaml.ScalarNode) and not node_value.style):
			best_style = False
		value.append((node_key, node_value))
	if flow_style is None:
		if dump.default_flow_style is not None:
			node.flow_style = dump.default_flow_style
		else:
			node.flow_style = best_style
	return node

##### Register

yaml.add_constructor(u'tag:yaml.org,2002:omap', _construct_odict)
yaml.constructor.SafeConstructor.add_constructor(u'tag:yaml.org,2002:omap', _construct_odict)
yaml.add_constructor(u'tag:yaml.org,2002:map', _construct_mapping)
yaml.constructor.SafeConstructor.add_constructor(u'tag:yaml.org,2002:map', _construct_mapping)

# Note: Register these two if behavior according to standard is desired, where OrderedDict
#       gets mapped to omap.
#yaml.representer.SafeRepresenter.add_representer(collections.OrderedDict, _repr_odict)
#yaml.add_representer(collections.OrderedDict, _repr_odict)

# Note: Register these two to map the OrderedDict to yaml mapping. Note that standard
#       conforming implementations are not required to keep the ordering when they
#       read the file.
yaml.representer.SafeRepresenter.add_representer(collections.OrderedDict, _repr_dict)
yaml.add_representer(collections.OrderedDict, _repr_dict)

if __name__ == '__main__':
	import textwrap

	sample = """
	one:
	  two: fish
	  red: fish
	  blue: fish
	two:
	  a: yes
	  b: no
	  c: null
	"""

	xsample = """
	!!omap
	- z: 1
	- y: 2
	- x: 3
	"""
	data = yaml.load(textwrap.dedent(sample))

	assert type(data) is collections.OrderedDict
	print data

	s2 = yaml.safe_dump(data)
	print s2
	
	data2 = yaml.load(s2)
	print data2
	assert data == data2

