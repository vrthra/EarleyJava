package parser;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
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

class SK {
    State s;
    char k;
    SK(State s, char k) {
        this.s = s;
        this.k = k;
    }
}

class VKC {
    SK sk;
    List<Column> chart;
    public VKC(SK sk, List<Column> chart) {
        this.sk = sk;
        this.chart = chart;
    }
}

class NamedForest {
    String name;
    ArrayList<ArrayList<VKC>> paths;
    public NamedForest(String name, ArrayList<ArrayList<VKC>> vkcs) {
        this.name = name;
        this.paths = vkcs;
    }
}

class SIInfo {
    State state;
    int index;
    char c;
    public SIInfo(State state, int index, char c) {
        this.state = state;
        this.index = index;
        this.c = c;
    }
}

class G {
    Grammar grammar;
    Map<String, Double> min_len;
    G(Grammar g) {
        this.grammar = g;
        this.min_len = this.compute_min_length();
    }

    public List<String> nullable() {
        List<String> nullable = new ArrayList<String>();
        for (String key : this.min_len.keySet()) {
            if (this.min_len.get(key) == 0.0) {
                nullable.add(key);
            }
        }
        return nullable;
    }

    private double _key_min_length(String k, Set<String> seen) {
        if (!this.grammar.containsKey(k)) {
            return k.length();
        }
        if (seen.contains(k)) {
            return Double.POSITIVE_INFINITY;
        }

        double min = 0;
        for (GRule r : this.grammar.get(k)) {
            Set<String> inter = new HashSet<String>(seen);
            inter.add(k);
            double m = this._rule_min_length(r, inter);
            if (m < min) {
                min = m;
            }
        }
        return min;
    }

    private double _rule_min_length(GRule rule, Set<String> seen) {
        double sum = 0;
        for (String k : rule) {
            sum += this._key_min_length(k, seen);
        }
        return sum;
    }

    Map<String, Double> compute_min_length() {
        Map<String, Double> min_len = new HashMap<String, Double>();
        for (String k : this.grammar.keySet()) {
            min_len.put(k, _key_min_length(k, new HashSet<String>()));
        }
        return min_len;
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
    String name;
    ArrayList<ParseTree> children;
    public ParseTree(String name, ArrayList<ParseTree> children) {
        this.name = name;
        this.children = children;
    }
}

class ParseForest implements Iterable<ParseTree> {
    int cursor;
    List<State> states;
    public ParseForest(int cursor, List<State> states) {
        this.cursor = cursor;
        this.states = states;
    }

    @Override
    public Iterator<ParseTree> iterator() {
        // TODO
        return null;
    }

}

interface ParserI {
    ParseForest parse_prefix(String text, String start_symbol);
    Iterator<ParseTree> parse(String text, String start_symbol) throws ParseException;
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

    public Iterator<ParseTree> parse(String text) throws ParseException {
        return this.parse(text, this.start_symbol);
    }

    @Override
    public Iterator<ParseTree> parse(String text, String start_symbol) throws ParseException {
        ParseForest p = this.parse_prefix(text, start_symbol);

        if (p.cursor < text.length()) {
            throw new ParseException("Syntax Error at: " + p.cursor);
        }
        return p.iterator();
    }
}

class Item {
    String name;
    GRule expr;
    int dot;
    Item(String name, GRule expr, int dot) {
        this.name = name;
        this.expr = expr;
        this.dot = dot;
    }

    public boolean finished() {
        return this.dot >= this.expr.size();
    }

    Item advance() {
        return new Item(this.name, this.expr, this.dot + 1);
    }

    public String at_dot() {
        if (this.dot < this.expr.size()) {
            return this.expr.get(this.dot);
        } else {
            return null;
        }
    }

}

class State extends Item {
    public Column s_col;
    public Column e_col;
    State(String name, GRule expr, int dot, Column s_col, Column e_col) {
        super(name, expr, dot);
        this.s_col = s_col;
        this.e_col = e_col;
    }

    String _idx(Column v) {
        if (v == null) return "-1";
        return Integer.toString(v.index);
    }

    public String toString() {
        ArrayList<String> lst = new ArrayList<String>();
        for (int i = 0; i < this.expr.size(); i++) {
            lst.add(this.expr.get(i).toString());
            if (i == this.dot) {
                lst.add("|");
            }
        }
        //lst = [ str(p) for p in [*this.expr[:this.dot], '|', *this.expr[this.dot:]] ]
        return this.name + ":= " + String.join(" ", lst) + _idx(this.s_col) + "," + _idx(this.e_col);
    }

    State copy() {
        return new State(this.name, this.expr, this.dot, this.s_col, this.e_col);
    }

    State advance() {
        return new State(this.name, this.expr, this.dot + 1, this.s_col, null);
    }

    State back() {
        return new TState(this.name, this.expr, this.dot - 1, this.s_col, this.e_col);
    }

    String _t() {
        return (this.name + "," + this.expr.toString() + "," + this.dot + "," + this.s_col.index);
    }

    boolean equals(State s) {
        return this._t() == s._t();
    }
}

class TState extends State {
    public TState(String name, GRule expr, int dot, Column s_col, Column e_col) {
        super(name, expr, dot, s_col, e_col);
    }
    TState copy() {
        return new TState(this.name, this.expr, this.dot, this.s_col, this.e_col);
    }
}

class Column {
    int index;
    String letter;
    ArrayList<State> states = new ArrayList<State>();
    HashMap<String, State> unique = new HashMap<String, State>();
    HashMap<String, State> transitives = new HashMap<String, State>();

    Column(int index, String letter) {
        this.index = index;
        this.letter = letter;
    }

    public String toString() {
        ArrayList<String> finished_states = new ArrayList<String>();
        for (State s: this.states) {
            if (s.finished()) {
                finished_states.add(s.toString());
            }
        }
        String finished = String.join("\n", finished_states);
        return String.format("%s chart[%d]\n%s", this.letter, this.index, finished);
    }

    State add(State state) {
        String s_state = state.toString();
        if (this.unique.containsKey(s_state) ) {
            return this.unique.get(s_state);
        }
        this.unique.put(s_state, state);
        this.states.add(state);
        state.e_col = this;
        return this.unique.get(s_state);
    }
    State add_transitive(String key, State state) {
        // assert key not in self.transitives
        this.transitives.put(key, new TState(state.name, state.expr, state.dot, state.s_col, state.e_col));
        return this.transitives.get(key);
    }
}
class EarleyParser extends Parser {
    List<String> epsilon;
    List<Column> table;
    private boolean log = false;
    public EarleyParser(Grammar grammar) {
        super(grammar);
        G g = new G(grammar);
        this.epsilon = g.nullable();
    }

    void predict(Column col, String sym,State state) {
        for (GRule alt : this.grammar.get(sym)) {
            col.add(new State(sym, alt, 0, col, null));
        }
        if (this.epsilon.contains(sym)) {
            col.add(state.advance());
        }
    }

    void scan(Column col, State state, String letter) {
        if (letter == col.letter) {
            col.add(state.advance());
        }
    }

    void complete(Column col, State state) {
        this.earley_complete(col, state);
    }

    void earley_complete(Column col, State state) {
        List<State> parent_states = new ArrayList<State>();
        for (State st: state.s_col.states) {
            if (st.at_dot() == state.name) {
                parent_states.add(st);
            }
        }
        for (State st : parent_states) {
            col.add(st.advance());
        }
    }

    List<Column> chart_parse(List<String> words, String start) {
        GRule alt = this.grammar.get(start).get(0);
        List<Column> chart = new ArrayList<Column>();
        chart.add(new Column(0, null));
        for (int i = 1; i < (words.size() + 1); i++) {
            String tok = words.get(i - 1);
            chart.add(new Column(i, tok));
        }
        State s = new State(start, alt, 0, chart.get(0), null);
        chart.get(0).add(s);
        return this.fill_chart(chart);
    }

    List<Column> fill_chart(List<Column> chart) {
        for (int i = 0; i < chart.size(); i++) {
            Column col = chart.get(i);
            for (State state: col.states) {
                if (state.finished()) {
                    this.complete(col, state);
                } else {
                    String sym = state.at_dot();
                    if (this.grammar.containsKey(sym)) {
                        this.predict(col, sym, state);
                    } else {
                        if (i + 1 >= chart.size()) {
                            continue;
                        }
                        this.scan(chart.get(i + 1), state, sym);
                    }
                }
            }

            if (this.log) {
                out(col.toString() + "\n");
            }
        }
        return chart;
    }


    private void out(String var) {
        System.out.println(var);
    }

    @Override
    public ParseForest parse_prefix(String text, String start_symbol) {
        this.table = this.chart_parse(Arrays.asList(text.split("")), start_symbol);
        List<State> states = new ArrayList<State>();
        for (int i = this.table.size(); i != 0; i--) {
            Column col = this.table.get(i-1);
            for (State st : col.states) {
                if (st.name.equals(start_symbol)) {
                    states.add(st);
                }
            }
            if (states.size() != 0) {
                return new ParseForest(col.index, states);
            }
        }
        return new ParseForest(-1, states);
    }

    ArrayList<ArrayList<SK>> _paths(int frm, List<Column> chart, State state, int start, char k, GRule e) {
        if (e.isEmpty()) {
            ArrayList<ArrayList<SK>> sk = new ArrayList<ArrayList<SK>>();
            if (start == frm) {
                ArrayList<SK> sk1 = new ArrayList<SK>();
                sk1.add(new SK(state, k));
                sk.add(sk1);
                return sk;
            } else {
                return sk;
            }
        } else {
            ArrayList<ArrayList<SK>> sk = new ArrayList<ArrayList<SK>>();
            for (ArrayList<SK> r : this.parse_paths(e, chart, frm, start)) {
                ArrayList<SK> sk1 = new ArrayList<SK>();
                sk1.add(new SK(state, k));
                sk1.addAll(r);
                sk.add(sk1);
            }
            return sk;
            // return [[(state, k)] + r for r in self.parse_paths(e, chart, frm, start)];
        }
    }

    ArrayList<ArrayList<SK>> parse_paths(GRule named_expr, List<Column> chart, int frm, int til) {
        ArrayList<SIInfo> starts = new ArrayList<SIInfo>();
        String var = named_expr.get(named_expr.size()-1);
        GRule expr = (GRule) named_expr.subList(0, named_expr.size() - 2);
        if (!this.grammar.containsKey(var)) {
            if (til > 0 && chart.get(til).letter.equals(var)) {
                starts.add(new SIInfo(new State(null, expr, 0, null, null), til - var.length(), 't'));
            }
        } else {
            for (State s: chart.get(til).states) {
                if (s.finished() && s.name.equals(var)) {
                    starts.add(new SIInfo(s, s.s_col.index, 'n'));
                }
            }
        }
        ArrayList<ArrayList<SK>> result = new ArrayList<ArrayList<SK>>();
        for (SIInfo sii : starts) {
            for (ArrayList<SK> pth : this._paths(frm, chart, sii.state, sii.index, sii.c, expr)) {
                result.add(pth);
            }
        }
        return result;
        //return [p for s, start, k in starts for p in _paths(frm, chart, s, start, k, named_expr.expr)];
    }

    NamedForest parse_forest(List<Column> chart, State state) {
        ArrayList<ArrayList<SK>> pathexprs = null;
        if (!state.expr.isEmpty()) {
            pathexprs = this.parse_paths(state.expr, chart, state.s_col.index, state.e_col.index);
        } else {
            pathexprs = new ArrayList<ArrayList<SK>>();
        }

        ArrayList<ArrayList<VKC>> vkcs = new ArrayList<ArrayList<VKC>>();
        for (ArrayList<SK> pathexpr: pathexprs) {
            ArrayList<VKC> vkc = new ArrayList<VKC>();
            for (int i = pathexpr.size(); i != 0; i--) {
                SK vk = pathexpr.get(i - 1);
                vkc.add(new VKC(vk, chart));
            }
            vkcs.add(vkc);
        }
        return new NamedForest(state.name, vkcs);
        /*return state.name, [[(v, k, chart) for v, k in reversed(pathexpr)]
                            for pathexpr in pathexprs];*/
    }

    NamedForest forest(State s, char kind, List<Column> chart) {
        if (kind == 'n') {
            return this.parse_forest(chart, s);
        } else {
             return new NamedForest(s.name, new ArrayList<ArrayList<VKC>>());
         }
    }

    ParseTree extract_a_tree(NamedForest forest_node) {
        if (forest_node.paths.isEmpty()) {
            return new ParseTree(forest_node.name, new ArrayList<ParseTree>());
        }

        ArrayList<ParseTree> lst = new ArrayList<ParseTree>();

		for (VKC p : forest_node.paths.get(0)) {
            lst.add(this.extract_a_tree(this.forest(p.sk.s, p.sk.k, p.chart)));
        }
        //[self.extract_a_tree(self.forest(*p)) for p in paths[0]]
        return new ParseTree(forest_node.name, lst);
    }


    Iterator<ParseTree> extract_trees(NamedForest forest) {
        ArrayList<ParseTree> pt = new ArrayList<ParseTree>();
        pt.add(this.extract_a_tree(forest));
        return pt.iterator();
    }

    public Iterator<ParseTree> parse(String text,String start_symbol) throws ParseException {
        ParseForest p = this.parse_prefix(text, start_symbol);
        State start = null;
        for (State s : p.states) {
            if (s.finished()) {
                start = s;
            }
        }

        if (p.cursor < text.length() || (start == null)) {
            throw new ParseException("at " + p.cursor);
        }

        NamedForest forest = this.parse_forest(this.table, start);
        return this.extract_trees(forest);
    }
/*




    def extract_trees(self, forest_node):
        name, paths = forest_node
        if not paths:
            yield (name, [])
        results = []
        for path in paths:
            ptrees = [self.extract_trees(self.forest(*p)) for p in path]
            for p in zip(*ptrees):
                yield (name, p)
*/
}

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

    def parse(self, text, start_symbol):
        cursor, states = self.parse_prefix(text, start_symbol)
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
