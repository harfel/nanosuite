import sys
from collections import namedtuple
from itertools import chain
from ply import lex
from ply.yacc import yacc

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
    r'(\d+\.\d*(e\-?\d+)?|inf)'
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

t_ignore = ' \t'      # pylint: disable=invalid-name

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"Illegal character '{t.value[0]}' in line {t.lineno}")
    t.lexer.skip(1)


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
    if len(p) == 4 and p[2] == '->':
        p[0] = Reaction(educts=p[1], products=p[3], rate=VarDef(None, 1))
    elif len(p) == 6:
        p[0] = Reaction(educts=p[1], products=p[3], rate=p[5])
    elif len(p) == 4 and p[2] == '<=>':
        p[0] = BalancedReaction(educts=p[1], products=p[3],
                                forward=VarDef(None, 1),backward=VarDef(None, 1))
    else:
        p[0] = BalancedReaction(educts=p[1], products=p[3],
                                forward=p[5],backward=p[7])

def p_reactants(p):
    """reactants : species
                 | species '+' reactants
                 | INT '*' species
                 | INT '*' species '+' reactants
                 | INT species
                 | INT species '+' reactants
    """
    if len(p) == 2:
        p[0] = [(p[1], 1)]
    elif len(p) == 4 and p[2] == '+':
        p[0] = [ (p[1], 1)] + p[3]
    elif len(p) == 4 and p[2] == '*':
        p[0] = [(p[3], p[1])]
    elif len(p) == 6:
        p[0] = [(p[3], p[1])] + p[5]
    elif len(p) == 3:
        p[0] = [(p[2], p[1])]
    else:
        p[0] = [(p[2], p[1])] + p[4]

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
               | LABEL '=' INT
               | LABEL
               | REAL
               | INT
    """
    if len(p) == 4:
        p[0] = VarDef(name=p[1], value=p[3])
    elif isinstance(p[1], float):
        p[0] = VarDef(name=None, value=p[1])
    else:
        p[0] = VarDef(name=p[1], value=1.)

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
        p[0] = FractionDef(p[1], p[3])
    else:
        p[0] = FractionDef(None, 0.)

def p_error(t):
    start = example.rfind('\n', 0, t.lexpos) + 1
    col = t.lexpos - start + 1
    sys.stderr.write(f"Syntax error: '{t.value}' in line {t.lineno}\n")
    sys.stderr.write(example[start: example.find('\n', t.lexpos)+1])
    sys.stderr.write(f"{(col-1)*' '}^\n")

# AST objects

VarDef = namedtuple('VarDef', ['name', 'value'])
FractionDef = namedtuple('FractionDef', ['suffix', 'fraction'])
SpeciesDef = namedtuple('SpeciesDef', ['species', 'subspecies', 'remains'])
Species  =namedtuple('Species', ['name', 'suffix'])
Reaction = namedtuple('Reaction', ['educts', 'products', 'rate'])
BalancedReaction = namedtuple('BalancedReaction',
                               ['educts', 'products', 'forward', 'backward'])
CrnDef = namedtuple('CrnDef', ['reactions', 'species_defs'])


# Example Usage

example = """
    probe contains burst with p_burst = 0.1,
          contains stagnating with p_stagnating = 0.2,
          rest pure 

    probe + input -> intermediate + output
    probe [burst] + input -> intermediate + output;     k=inf
    probe [stagnating] + input <=> stagnation + output

"""

alternative_example = """
    probe + input -> intermediate + output

    probe contains burst with p_burst = 0.1
    probe [burst] + input -> intermediate + output; k=inf

    probe contains stagnating with p_stagnating = 0.2
    probe [stagnating] + input <=> stagnation + output
"""


lexer = lex.lex()
parser = yacc()

ast = parser.parse(example, lexer=lexer, )

# TODO: collect all species suffixes

# homogenize species names
for reaction in ast.reactions:
    for species, _ in chain(reaction.educts, reaction.products):
        if species.suffix != None:
            continue
        # FIXME:
        # elif species.name in ast.species_defs:
        #     species.suffix = ast.species_defs[species.name].remains or 'rest'

# TODO: name unnamed variables
# TODO: build flat CRN
# TODO: burst integration
# TODO: integration
# TODO: compact CRN
