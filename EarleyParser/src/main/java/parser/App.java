package parser;

/**
 * Earley Parser
 *
 */
public class App 
{
    public static void main(String[] args) {
        ParserLib pl = new ParserLib(args[0]);
        ParseTree result = pl.parse_text(args[1]);
        pl.show_tree(result);
    }
}
