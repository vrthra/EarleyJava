package parser;

import java.io.IOException;

/**
 * Earley Parser
 *
 */
public class App {
    public static void main(String[] args) {
        ParserLib pl;
        try {
            pl = new ParserLib(args[0]);
            ParseTree result = pl.parse_text(args[1]);
            System.out.println(pl.get_json(result).toString(4));
        } catch (ParseException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
