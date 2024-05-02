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
import sys
import re
from collections import namedtuple
import lmfit # type: ignore
from ply import lex # type: ignore
from ply.yacc import yacc # type: ignore

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

t_ignore = ' \t'        # pylint: disable=invalid-name

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    """Handle syntax errors on the lexer level"""
    start = example.rfind('\n', 0, t.lexpos) + 1 # FIXME: do not use example!
    col = t.lexpos - start + 1
    sys.stderr.write(f"Syntax error: '{t.value[0]}' in line {t.lineno}\n")
    sys.stderr.write(example[start: example.find('\n', t.lexpos)+1])
    sys.stderr.write(f"{(col-1)*' '}^\n")
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
    if len(p) == 4 and p[2] == t_RIGHT_ARROW:
        name = f'_k_{len(p.parser.context)+1}'
        p.parser.context[name] = VarDef(name, None)
        p[0] = Reaction(educts=p[1], products=p[3], rate=name)
    elif len(p) == 6:
        p[0] = Reaction(educts=p[1], products=p[3], rate=p[5])
    elif len(p) == 4 and p[2] == t_DOUBLE_ARROW:
        fname = f'_k_{len(p.parser.context)+1}'
        bname = f'_k_{len(p.parser.context)+2}'
        p.parser.context[fname] = VarDef(fname, None)
        p.parser.context[bname] = VarDef(bname, None)
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
    p[0] = p[1] # FIXME: later this must include labels
    return

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
        if p[1].startswith('_'):
            raise ValueError(f"Rate constant not allowed to start with underscore: {p[1]}.")
        if p[1] in p.parser.context and p.parser.context[p[1]].value != p[3]:
            raise ValueError("Inconsistent values for rate constant {p[1]}.")
        p.parser.context[p[1]] = VarDef(p[1], p[3])
        p[0] = p[1]
    elif isinstance(p[1], (int, float)):
        name = f'_k_{len(p.parser.context)+1}'
        p.parser.context[name] = VarDef(name, p[1])
        p[0] = name
    else:
        if p[1].startswith('_'):
            raise ValueError(f"Rate constant not allowed to start with underscore: {name}.")
        if p[1] not in p.parser.context:
            p.parser.context[p[1]] = VarDef(p[1], None)
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
        p[0] = FractionDef(p[1], p[3])
    else:
        p[0] = FractionDef(None, 0.)

def p_error(t):
    """Handle errors on the grammar level"""
    start = example.rfind('\n', 0, t.lexpos) + 1 # FIXME: do not use example!
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


def parse(string: str) -> CrnDef:
    """Construct abstract CrnDef from string input"""
    parser = yacc()
    parser.context = {}
    crn_def = parser.parse(string, lexer=lex.lex())

    # replace rate name placeholders with descriptive variable names
    pattern = re.compile(r'\d+')
    bound_nums = [match for name, param in parser.context.items()
                  for match in pattern.findall(name)
                  if not name.startswith('_')]
    free_nums = [idx_str for idx, _ in enumerate(parser.context, 1)
                 if (idx_str:=str(idx)) not in bound_nums]
    for idx, reaction in enumerate(crn_def.reactions):
        if isinstance(reaction, Reaction):
            if not reaction.rate.startswith('_'):
                continue
            parser.context[reaction.rate] = VarDef(f'k{free_nums.pop(0)}',
                                                   parser.context[reaction.rate].value)
        else:
            if not reaction.forward.startswith('_'):
                continue
            num = free_nums.pop(0)
            parser.context[reaction.forward] = VarDef(f'kf{num}',
                                                      parser.context[reaction.forward].value)
            parser.context[reaction.backward] = VarDef(f'kb{num}',
                                                       parser.context[reaction.backward].value)

    # set default parameter values
    params = {
        key: lmfit.Parameter(name, val or 1., min=0.)
        for key, (name, val) in parser.context.items()
    }

    crn_def = CrnDef([
        Reaction(rct.educts, rct.products, params[rct.rate])
        if isinstance(rct, Reaction) else
        BalancedReaction(rct.educts, rct.products,
                         params[rct.forward], params[rct.backward])
        for rct in crn_def.reactions
    ], crn_def.species_defs)

    return crn_def


if __name__ == '__main__':
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

    parse(example)
