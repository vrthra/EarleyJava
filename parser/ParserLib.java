package parser;

import java.util.HashMap;
import java.util.Iterator;

import java.util.ArrayList;
import java.util.Arrays;

class GRule extends ArrayList<String> {
    public GRule() {
    }
}

class GDef extends ArrayList<GRule> {
    public GDef() {
    }
}

class Grammar extends HashMap<String, GDef> {
    public Grammar() {
    }
}

public class ParserLib {
    static Grammar single_char_tokens(Grammar grammar) {
        Grammar g_ = new Grammar();
        for (String key : grammar.keySet()) {
            GDef rules_ = new GDef();
            for (GRule rule : grammar.get(key)) {
                GRule rule_ = new GRule();
                for (String token : rule) {
                    if (grammar.keySet().contains(token)) {
                        rule_.add(token);
                    } else {
                        for (String c : token.split("")) {
                            rule_.add(c);
                        }
                    }
                }
                rules_.add(rule_);
            }
            g_.put(key, rules_);
        }
        return g_;
    }

    public static void main(String[] args) {
        System.out.println("Hello");
    }
}
// Parser.py

class ParseTree {
}

class ParseForest implements Iterable<ParseTree> {
    int cursor;

    @Override
    public Iterator<ParseTree> iterator() {
        return null;
    }

}

interface ParserI {
    ParseForest parse_prefix(String text);
    Iterator<ParseTree> parse(String text) throws ParseException;
}

abstract class Parser implements ParserI {
    String start_symbol;
    Grammar grammar;
    Parser(Grammar grammar) {
        this(grammar, "<start>");
    }
    Parser(Grammar grammar, String start_symbol) {
        this.start_symbol = start_symbol;
        this.grammar = ParserLib.single_char_tokens(grammar);
        // we do not require a single rule for the start symbol
        int start_rules_len = grammar.get(start_symbol).size();
        if (start_rules_len != 1) {
            GRule gr = new GRule();
            gr.addAll(Arrays.asList(new String[]{this.start_symbol}));
            GDef gd = new GDef();
            gd.addAll(Arrays.asList(new GRule[]{gr}));
            this.grammar.put("<>", gd);
        }
    }

    @Override
    public Iterator<ParseTree> parse(String text) throws ParseException {
        ParseForest p = this.parse_prefix(text);

        if (p.cursor < text.length()) {
            throw new ParseException("Syntax Error at: " + p.cursor);
        }
        return p.iterator();
    }

    public Iterator<ParseTree> parse_on(String text, String start_symbol) throws ParseException {
        String old_start = this.start_symbol;
        try {
            this.start_symbol = start_symbol;
            return this.parse(text);
        } finally {
            // TODO. This is not what we want. We want this entire parse to happen
            // on start_symbol, and at the end reset the object back to old_start
            this.start_symbol = old_start;
        }
    }
}

/*
class Column:
    def __str__(self):
        return "%s chart[%d]\n%s" % (self.letter, self.index, "\n".join(
            str(state) for state in self.states if state.finished()))

    def add(self, state):
        if state in self._unique:
            return self._unique[state]
        self._unique[state] = state
        self.states.append(state)
        state.e_col = self
        return self._unique[state]

    def __init__(self, index, letter):
        self.index, self.letter = index, letter
        self.states, self._unique, self.transitives = [], {}, {}

    def add_transitive(self, key, state):
        assert key not in self.transitives
        self.transitives[key] = state
        return self.transitives[key]

    def add_transitive(self, key, state):
        assert key not in self.transitives
        self.transitives[key] = TState(state.name, state.expr, state.dot,
                                       state.s_col, state.e_col)
        return self.transitives[key]


class Item:
    def __init__(self, name, expr, dot):
        self.name, self.expr, self.dot = name, expr, dot

    def finished(self):
        return self.dot >= len(self.expr)

    def advance(self):
        return Item(self.name, self.expr, self.dot + 1)

    def at_dot(self):
        return self.expr[self.dot] if self.dot < len(self.expr) else None

class State(Item):
    def __init__(self, name, expr, dot, s_col, e_col=None):
        super().__init__(name, expr, dot)
        self.s_col, self.e_col = s_col, e_col

    def __str__(self):
        def idx(var):
            return var.index if var else -1

        return self.name + ':= ' + ' '.join([
            str(p)
            for p in [*self.expr[:self.dot], '|', *self.expr[self.dot:]]
        ]) + "(%d,%d)" % (idx(self.s_col), idx(self.e_col))

    def copy(self):
        return State(self.name, self.expr, self.dot, self.s_col, self.e_col)

    def _t(self):
        return (self.name, self.expr, self.dot, self.s_col.index)

    def __hash__(self):
        return hash(self._t())

    def __eq__(self, other):
        return self._t() == other._t()

    def advance(self):
        return State(self.name, self.expr, self.dot + 1, self.s_col)

    def back(self):
        return TState(self.name, self.expr, self.dot - 1, self.s_col, self.e_col)
*/
/*
def fixpoint(f):
    def helper(arg):
        while True:
            sarg = str(arg)
            arg_ = f(arg)
            if str(arg_) == sarg:
                return arg
            arg = arg_

    return helper

def rules(grammar):
    return [(key, choice)
            for key, choices in grammar.items()
            for choice in choices]

def terminals(grammar):
    return set(token
               for key, choice in rules(grammar)
               for token in choice if token not in grammar)

def nullable_expr(expr, nullables):
    return all(token in nullables for token in expr)

def nullable(grammar):
    productions = rules(grammar)

    @fixpoint
    def nullable_(nullables):
        for A, expr in productions:
            if nullable_expr(expr, nullables):
                nullables |= {A}
        return (nullables)

    return nullable_({EPSILON})
*/


/*
class EarleyParser(Parser):
    def chart_parse(self, words, start):
        alt = tuple(*self.grammar[start])
        chart = [Column(i, tok) for i, tok in enumerate([None, *words])]
        chart[0].add(State(start, alt, 0, chart[0]))
        return self.fill_chart(chart)


    def predict(self, col, sym, state):
        for alt in self.grammar[sym]:
            col.add(State(sym, tuple(alt), 0, col))


    def scan(self, col, state, letter):
        if letter == col.letter:
            col.add(state.advance())


    def complete(self, col, state):
        return self.earley_complete(col, state)

    def earley_complete(self, col, state):
        parent_states = [
            st for st in state.s_col.states if st.at_dot() == state.name
        ]
        for st in parent_states:
            col.add(st.advance())


    def fill_chart(self, chart):
        for i, col in enumerate(chart):
            for state in col.states:
                if state.finished():
                    self.complete(col, state)
                else:
                    sym = state.at_dot()
                    if sym in self.grammar:
                        self.predict(col, sym, state)
                    else:
                        if i + 1 >= len(chart):
                            continue
                        self.scan(chart[i + 1], state, sym)
            if self.log:
                print(col, '\n')
        return chart


    def parse_prefix(self, text):
        self.table = self.chart_parse(text, self.start_symbol())
        for col in reversed(self.table):
            states = [
                st for st in col.states if st.name == self.start_symbol()
            ]
            if states:
                return col.index, states
        return -1, []

    def parse(self, text):
        cursor, states = self.parse_prefix(text)
        start = next((s for s in states if s.finished()), None)

        if cursor < len(text) or not start:
            raise SyntaxError("at " + repr(text[cursor:]))

        forest = self.parse_forest(self.table, start)
        for tree in self.extract_trees(forest):
            yield tree

    def parse_paths(self, named_expr, chart, frm, til):
        def paths(state, start, k, e):
            if not e:
                return [[(state, k)]] if start == frm else []
            else:
                return [[(state, k)] + r
                        for r in self.parse_paths(e, chart, frm, start)]

        *expr, var = named_expr
        starts = None
        if var not in self.grammar:
            starts = ([(var, til - len(var),
                        't')] if til > 0 and chart[til].letter == var else [])
        else:
            starts = [(s, s.s_col.index, 'n') for s in chart[til].states
                      if s.finished() and s.name == var]

        return [p for s, start, k in starts for p in paths(s, start, k, expr)]


    def forest(self, s, kind, chart):
        return self.parse_forest(chart, s) if kind == 'n' else (s, [])

    def parse_forest(self, chart, state):
        pathexprs = self.parse_paths(state.expr, chart, state.s_col.index,
                                     state.e_col.index) if state.expr else []
        return state.name, [[(v, k, chart) for v, k in reversed(pathexpr)]
                            for pathexpr in pathexprs]

    def extract_a_tree(self, forest_node):
        name, paths = forest_node
        if not paths:
            return (name, [])
        return (name, [self.extract_a_tree(self.forest(*p)) for p in paths[0]])

    def extract_trees(self, forest):
        yield self.extract_a_tree(forest)

    def extract_trees(self, forest_node):
        name, paths = forest_node
        if not paths:
            yield (name, [])
        results = []
        for path in paths:
            ptrees = [self.extract_trees(self.forest(*p)) for p in path]
            for p in zip(*ptrees):
                yield (name, p)

    def __init__(self, grammar, **kwargs):
        super().__init__(grammar, **kwargs)
        self.epsilon = nullable(self.grammar)

    def predict(self, col, sym, state):
        for alt in self.grammar[sym]:
            col.add(State(sym, tuple(alt), 0, col))
        if sym in self.epsilon:
            col.add(state.advance())
            
*/
/*
class TState(State):
    def copy(self):
        return TState(self.name, self.expr, self.dot, self.s_col, self.e_col)
*/
/*
class LeoParser(EarleyParser):
    def complete(self, col, state):
        return self.leo_complete(col, state)

    def leo_complete(self, col, state):
        detred = self.deterministic_reduction(state)
        if detred:
            col.add(detred.copy())
        else:
            self.earley_complete(col, state)

    def deterministic_reduction(self, state):
        raise NotImplemented()

    def uniq_postdot(self, st_A):
        col_s1 = st_A.s_col
        parent_states = [
            s for s in col_s1.states if s.expr and s.at_dot() == st_A.name
        ]
        if len(parent_states) > 1:
            return None
        matching_st_B = [s for s in parent_states if s.dot == len(s.expr) - 1]
        return matching_st_B[0] if matching_st_B else None

    def get_top(self, state_A):
        st_B_inc = self.uniq_postdot(state_A)
        if not st_B_inc:
            return None

        t_name = st_B_inc.name
        if t_name in st_B_inc.e_col.transitives:
            return st_B_inc.e_col.transitives[t_name]

        st_B = st_B_inc.advance()

        top = self.get_top(st_B) or st_B
        return st_B_inc.e_col.add_transitive(t_name, top)

    def deterministic_reduction(self, state):
        return self.get_top(state)

    def __init__(self, grammar, **kwargs):
        super().__init__(grammar, **kwargs)
        self._postdots = {}

    def uniq_postdot(self, st_A):
        col_s1 = st_A.s_col
        parent_states = [
            s for s in col_s1.states if s.expr and s.at_dot() == st_A.name
        ]
        if len(parent_states) > 1:
            return None
        matching_st_B = [s for s in parent_states if s.dot == len(s.expr) - 1]
        if matching_st_B:
            self._postdots[matching_st_B[0]._t()] = st_A
            return matching_st_B[0]
        return None

    def expand_tstate(self, state, e):
        if state._t() not in self._postdots:
            return
        c_C = self._postdots[state._t()]
        e.add(c_C.advance())
        self.expand_tstate(c_C.back(), e)

    def rearrange(self, table):
        f_table = [Column(c.index, c.letter) for c in table]
        for col in table:
            for s in col.states:
                f_table[s.s_col.index].states.append(s)
        return f_table

    def parse(self, text):
        cursor, states = self.parse_prefix(text)
        start = next((s for s in states if s.finished()), None)
        if cursor < len(text) or not start:
            raise SyntaxError("at " + repr(text[cursor:]))

        self.r_table = self.rearrange(self.table)
        forest = self.extract_trees(self.parse_forest(self.table, start))
        for tree in forest:
            yield tree

    def parse_forest(self, chart, state):
        if isinstance(state, TState):
            self.expand_tstate(state.back(), state.e_col)

        return super().parse_forest(chart, state)
*/

/*
class IterativeEarleyParser(LeoParser):
    def parse_paths(self, named_expr_, chart, frm, til_):
        return_paths = []
        path_build_stack = [(named_expr_, til_, [])]

        def iter_paths(path_prefix, path, start, k, e):
            x = path_prefix + [(path, k)]
            if not e:
                return_paths.extend([x] if start == frm else [])
            else:
                path_build_stack.append((e, start, x))

        while path_build_stack:
            named_expr, til, path_prefix = path_build_stack.pop()
            *expr, var = named_expr

            starts = None
            if var not in self.grammar:
                starts = ([(var, til - len(var),
                        't')] if til > 0 and chart[til].letter == var else [])
            else:
                starts = [(s, s.s_col.index, 'n') for s in chart[til].states
                      if s.finished() and s.name == var]

            for s, start, k in starts:
                iter_paths(path_prefix, s, start, k, expr)

        return return_paths

    def choose_a_node_to_explore(self, node_paths, level_count):
        first, *rest = node_paths
        return first

    def extract_a_tree(self, forest_node_):
        start_node = (forest_node_[0], [])
        tree_build_stack = [(forest_node_, start_node[-1], 0)]

        while tree_build_stack:
            forest_node, tree, level_count = tree_build_stack.pop()
            name, paths = forest_node

            if not paths:
                tree.append((name, []))
            else:
                new_tree = []
                current_node = self.choose_a_node_to_explore(paths, level_count)
                for p in reversed(current_node):
                    new_forest_node = self.forest(*p)
                    tree_build_stack.append((new_forest_node, new_tree, level_count + 1))
                tree.append((name, new_tree))

        return start_node

    def extract_trees(self, forest):
        yield self.extract_a_tree(forest)
*/
/*
    test_cases = [
        (A1_GRAMMAR, '1-2-3+4-5'),
        # (A2_GRAMMAR, '1+2'),
        # (A3_GRAMMAR, '1+2+3-6=6-1-2-3'),
        # (LR_GRAMMAR, 'aaaaa'),
        # (RR_GRAMMAR, 'aa'),
        # (DIRECTLY_SELF_REFERRING, 'select a from a'),
        # (INDIRECTLY_SELF_REFERRING, 'select a from a'),
        # (RECURSION_GRAMMAR, 'AA'),
        # (RECURSION_GRAMMAR, 'AAaaaa'),
        # (RECURSION_GRAMMAR, 'BBccbb')
    ]

    for i, (grammar, text) in enumerate(test_cases):
        print(i, text)
        tree, *_ =  IterativeEarleyParser(grammar, canonical=True).parse(text)
        assert text == tree_to_string(tree)
*/
