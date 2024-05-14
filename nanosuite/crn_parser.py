"""Domain specific language parser for chemical reactio networks

    The format of the CRN specification language is as follows:
    each reaction is specified on a single line. A hash character (#)
    anywhere in the input marks the beginning of a comment that extends
    until the end of the line.

    Each line specifies an irreversible ('->') or reversible ('<=>')
    reaction among educts and products. Educts and products use '+' to
    separate individual chemical species names. Species names can be
    preceeded by their stoichiometric factor, optionally separated by
    a multiplicaton sign '*'.

    Reactions can be followed by a semicolon (;) after which the
    reaction rate constants are specified in the format name=value.
    For reversible reactions, the forward and backward constant
    specifications are separated by a comma. In both cases, both name
    and value are optional. If no value is given, 1 is assumed.
    If no name is given, a generic name that is not used in other
    reactions is provided.

    Reaction rates can be specified with a value of 'inf' which implies
    that they occur essentally instantaneously at the start of a
    simulation.

    If any chemical species name is followed by a label in square
    brackets, it is assumed to apply to a subspecies fraction of that
    species.

    Subspecies can be defined (at the beginning or anywhere) within
    the CRN definition as follows:
    
    SPECIES contains SUBSPECIES with VARIABLE=FRACTION,
            contains [...],
            rest SUBSPECIES

    For example:

    mRNA contains mutation with p=1e-4 rest wildtype

    Most of the subspecies definition is optional, so that subspecies
    can be defined without explicit fraction or variable name. If no
    explicit name is given for the rest, 'pure' is assumed.
"""
import re
from collections import namedtuple
from itertools import chain
import lmfit # type: ignore
from ply import lex # type: ignore
from ply.yacc import yacc # type: ignore


class CRNError(ValueError):
    """Exceptions for syntax errors in parser input"""


# Tokens

reserved = {
   'contains' : 'CONTAINS',
   'rest' : 'REST',
   'with' : 'WITH',
}

tokens = [
    'COMMENT',
    'RIGHT_ARROW',
    'DOUBLE_ARROW',
    'INT',
    'REAL',
    'LABEL',
] + list(reserved.values())

literals = list(';=+[],')

def t_COMMENT(_):       # pylint: disable=invalid-name
    r'\#.*'

def t_REAL(t):          # pylint: disable=invalid-name
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|inf'
    t.value = float(t.value)
    return t

def t_INT(t):           # pylint: disable=invalid-name
    r'\d+'
    t.value = int(t.value)
    return t

def t_LABEL(t):         # pylint: disable=invalid-name
    r'[a-zA-Z_0-9]+'
    t.type = reserved.get(t.value,'LABEL')
    return t

t_DOUBLE_ARROW = '<=>'  # pylint: disable=invalid-name
t_RIGHT_ARROW = '->'    # pylint: disable=invalid-name

t_ignore = ' \t'        # pylint: disable=invalid-name

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    """Handle syntax errors on the lexer level"""

    start = t.lexer.text.rfind('\n', 0, t.lexpos) + 1
    end = t.lexer.text.find('\n', t.lexpos)
    col = t.lexpos - start + 1
    msg = f"""Illegal character: '{t.value[0]}' in line {t.lineno}:
{t.lexer.text[start: end if end != -1 else None]}
{(col-1)*' '}^"""
    raise CRNError(msg)


# Grammar

def p_crn(p):
    """crn : statement
           | statement crn
    """
    species_defs = {} if len(p) == 2 else dict(p[2].species_defs)
    if isinstance(p[1], (Reaction, BalancedReaction)):
        if len(p) == 2:
            p[0] = CrnDef([p[1]], species_defs)
        else:
            p[0] = CrnDef([p[1]]+p[2].reactions, species_defs)
    else:
        if len(p) == 2:
            p[0] = CrnDef([], {p[1].species: p[1]})
        else:
            if p[1].species in species_defs:
                species_defs[p[1].species].subspecies.update(p[1].subspecies)
                if p[1].remains:
                    species_defs[p[1].species].remains = p[1].remains
            else:
                species_defs[p[1].species] = p[1]
            p[0] = CrnDef(p[2].reactions, species_defs)

def p_statement(p):
    """statement : COMMENT
                 | reaction
                 | species_def
    """
    p[0] = p[1]

def p_reaction(p):
    """reaction : reactants RIGHT_ARROW reactants
                | reactants RIGHT_ARROW reactants ';' var_def
                | reactants DOUBLE_ARROW reactants
                | reactants DOUBLE_ARROW reactants ';' var_def ',' var_def
    """
    if len(p) == 4 and p[2] == t_RIGHT_ARROW:
        name = f'_k_{len(p.parser.context)+1}'
        p.parser.context[name] = lmfit.Parameter(name, value=1, min=0)
        p[0] = Reaction(educts=p[1], products=p[3], rate=name)
    elif len(p) == 6:
        p[0] = Reaction(educts=p[1], products=p[3], rate=p[5])
    elif len(p) == 4 and p[2] == t_DOUBLE_ARROW:
        fname = f'_k_{len(p.parser.context)+1}'
        bname = f'_k_{len(p.parser.context)+2}'
        p.parser.context[fname] = lmfit.Parameter(fname, value=1, min=0)
        p.parser.context[bname] = lmfit.Parameter(bname, value=1, min=0)
        p[0] = BalancedReaction(educts=p[1], products=p[3], forward=fname, backward=bname)
    else:
        p[0] = BalancedReaction(educts=p[1], products=p[3], forward=p[5], backward=p[7])

def p_reactants(p):
    """reactants : species
                 | species '+' reactants
                 | INT '*' species
                 | INT '*' species '+' reactants
                 | INT species
                 | INT species '+' reactants
    """
    if len(p) == 2:
        p[0] = ((p[1], 1),)
    elif len(p) == 4 and p[2] == '+':
        p[0] = ((p[1], 1),) + p[3]
    elif len(p) == 4 and p[2] == '*':
        p[0] = ((p[3], p[1]),)
    elif len(p) == 6:
        p[0] = ((p[3], p[1]),) + p[5]
    elif len(p) == 3:
        p[0] = ((p[2], p[1]),)
    else:
        p[0] = ((p[2], p[1]),) + p[4]

def p_species(p):
    """species : LABEL
               | LABEL '[' LABEL ']'
    """
    if len(p) == 5:
        p[0] = Species(p[1], p[3])
    else:
        p[0] = Species(p[1], None)

def p_var_def(p):
    """var_def : LABEL '=' REAL
               | LABEL
               | REAL
    """
    if len(p) == 4:
        if p[1].startswith('_'):
            raise ValueError(f"Rate constant not allowed to start with underscore: {p[1]}.")
        if p[1] in p.parser.context and p.parser.context[p[1]].value != p[3]:
            raise ValueError("Inconsistent values for rate constant {p[1]}.")
        p.parser.context[p[1]] = lmfit.Parameter(p[1], value=p[3], min=0)
        p[0] = p[1]
    elif isinstance(p[1], float):
        name = f'_k_{len(p.parser.context)+1}'
        p.parser.context[name] = lmfit.Parameter(name, value=p[1], min=0)
        p[0] = name
    else:
        if p[1].startswith('_'):
            raise ValueError(f"Rate constant not allowed to start with underscore: {name}.")
        if p[1] not in p.parser.context:
            p.parser.context[p[1]] = lmfit.Parameter(p[1], value=1, min=0)
        p[0] = p[1]

def p_species_def(p):
    """species_def : LABEL subspecies_list
    """
    p[0] = SpeciesDef(p[1], p[2].subspecies, p[2].remains)

def p_subspecies_list(p):
    """
    subspecies_list : subspecies_def
                    | subspecies_def ',' subspecies_list
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = SpeciesDef(None, p[1].subspecies | p[3].subspecies, p[1].remains or p[3].remains)

def p_subspecies_def(p):
    """subspecies_def : CONTAINS fraction_def
                      | REST LABEL
    """
    if p[1] == 'contains':
        p[0] = SpeciesDef(None, {p[2].suffix: p[2].fraction}, None)
    else:
        p[0] = SpeciesDef(None, {}, p[2])

def p_fraction_def(p):
    """fraction_def : LABEL
                    | LABEL WITH var_def
    """
    if len(p) == 4:
        p.parser.context[p[3]].max = 1
        p[0] = FractionDef(p[1], p[3])
    else:
        name = f'_p_{len(p.parser.context)+1}'
        p.parser.context[name] = lmfit.Parameter(name, 0., min=0, max=1)
        p[0] = FractionDef(p[1], name)

def p_error(t):
    """Handle errors on the grammar level"""
    if not t:
        raise CRNError("Unexpected end of input")

    start = t.lexer.text.rfind('\n', 0, t.lexpos) + 1
    end = t.lexer.text.find('\n', t.lexpos)
    col = t.lexpos - start + 1
    msg = f"""Syntax error: '{t.value}' in line {t.lineno}:
{t.lexer.text[start: end if end != -1 else None]}
{(col-1)*' '}^"""
    raise CRNError(msg)


# AST objects

FractionDef = namedtuple('FractionDef', ['suffix', 'fraction'])
SpeciesDef = namedtuple('SpeciesDef', ['species', 'subspecies', 'remains'])
Species  =namedtuple('Species', ['name', 'suffix'])
Reaction = namedtuple('Reaction', ['educts', 'products', 'rate'])
BalancedReaction = namedtuple('BalancedReaction',
                               ['educts', 'products', 'forward', 'backward'])
CrnDef = namedtuple('CrnDef', ['reactions', 'species_defs'])


def replace_name_placeholders(crn_def, variables):
    """Substitute preliminary generated variable names"""
    pattern = re.compile(r'\d+')
    bound_nums = [match for name, param in variables.items()
                  for match in pattern.findall(name)
                  if not name.startswith('_')]
    free_nums = [idx_str for idx, _ in enumerate(variables, 1)
                  if (idx_str:=str(idx)) not in bound_nums]

    for reaction in crn_def.reactions:
        if isinstance(reaction, Reaction):
            if not reaction.rate.startswith('_'):
                continue
            variables[reaction.rate].name = f'k{free_nums.pop(0)}'
        else:
            if not reaction.forward.startswith('_'):
                continue
            num = free_nums.pop(0)
            variables[reaction.forward].name = f'kf{num}'
            variables[reaction.backward].name = f'kb{num}'

    for species_def in crn_def.species_defs.values():
        for suffix, rate in species_def.subspecies.items():
            if not rate.startswith('_'):
                continue
            variables[rate].name = f'p_{species_def.species}_{suffix}'

def parse(string: str) -> CrnDef:
    """Construct abstract CrnDef from string input"""
    lexer = lex.lex()
    lexer.text = string
    parser = yacc()
    parser.context = {}
    crn_def = parser.parse(string, lexer=lexer)

    # collect implicitly defined subspecies
    for reaction in crn_def.reactions:
        for reactant, _ in chain(reaction.educts, reaction.products):
            if not reactant.suffix:
                continue
            if reactant.name not in crn_def.species_defs:
                crn_def.species_defs[reactant.name] = SpeciesDef(reactant.name, {}, remains=None)
            if reactant.suffix not in crn_def.species_defs[reactant.name].subspecies:
                name = f'_p_{len(parser.context)+1}'
                parser.context[name] = lmfit.Parameter(name, min=0, max=1)
                crn_def.species_defs[reactant.name].subspecies[reactant.suffix] = name

    # replace rate name placeholders with descriptive variable names
    replace_name_placeholders(crn_def, parser.context)

    def stratify_species(reactants):
        return tuple(
            (f'{species.name}_'
             +f'{species.suffix or crn_def.species_defs[species.name].remains or "pure"}'
             if species.name in crn_def.species_defs else species.name, stoich)
            for species, stoich in reactants
        )

    crn_def = CrnDef([Reaction(stratify_species(rct.educts), stratify_species(rct.products),
                               parser.context[rct.rate])
                      if isinstance(rct, Reaction) else
                      BalancedReaction(stratify_species(rct.educts), stratify_species(rct.products),
                                       parser.context[rct.forward], parser.context[rct.backward])
                      for rct in crn_def.reactions],
                     {species: SpeciesDef(species,
                                          {f'{species}_{suffix}': parser.context[frac]
                                           for suffix, frac in spdef.subspecies.items()},
                                          f'{species}_{spdef.remains or "pure"}')
                      for species, spdef in crn_def.species_defs.items()})

    return crn_def
