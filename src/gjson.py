import json
json_grammar = { "<start>": [
    ["<json>"]
    ],
    "<json>": [
        ["<element>"]
        ],
    "<value>": [
        ["<object>"],
        ["<array>"],
        ["<string>"],
        ["<number>"],
        ["true"],
        ["false"],
        ["null"]
        ],
    "<object>": [
        ["{", "<ws>", "}"],
        ["{", "<members>", "}"]
        ],
    "<members>": [
        ["<member>"],
        ["<member>", ",", "<members>"]
        ],
    "<member>": [
        ["<ws>", "<string>", "<ws>", ":", "<element>"]
        ],
    "<array>": [
        ["[", "<ws>", "]"],
        ["[", "<elements>", "]"]
        ],
    "<elements>": [
        ["<element>"],
        ["<element>", ",", "<elements>"]
        ],
    "<element>": [
        ["<ws>", "<value>", "<ws>"]
        ],
    "<string>": [
        ["\"", "<characters>", "\""]
        ],
    "<characters>": [
        [""],
        ["<character>", "<characters>"]],
    "<character>": (
        [
            [chr(i)] for i in range(0x0020,0x10FFFF + 1)
            #[chr(i)] for i in range(0x20,0xFF + 1)
            if chr(i) not in {'"', '\\'}]
        + [["\\", "<escape>"]]),
    "<escape>": [
        ["\""],
        ["\\"],
        ["/"],
        ["b"],
        ["f"],
        ["n"],
        ["r"],
        ["t"],
        ["u", "<hex>", "<hex>", "<hex>", "<hex>"],
        ],
    "<hex>": ([["<digit>"]] + 
            [[chr(i)] for i in range(ord('A'), ord('F') + 1)] + 
            [[chr(i)] for i in range(ord('a'), ord('f') + 1)]),
    "<number>": [
            ["<integer>", "<fraction>", "<exponent>"]
            ],
    "<integer>": [
            ["<digit>"],
            ["<onenine>", "<digits>"],
            ["-", "<digits>"],
            ["-", "<onenine>", "<digits>"]
            ],
    "<digits>": [
            ["<digit>"],
            ["<digit>", "<digits>"]],
    "<digit>": [
            ["0"],
            ["<onenine>"]
            ],
    "<onenine>": [
            ["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"]],
    "<fraction>": [
            [""],
            [".", "<digits>"]
            ],
    "<exponent>": [
            [""],
            ["E", "<sign>", "<digits>"],
            ["e", "<sign>", "<digits>"]
            ],
    "<sign>": [
            [""],
            ["+"],
            ["-"]
            ],
    "<ws>": [
            [""],
            ["<sp1>", "<ws>"]],
    "<sp1>": [
            [chr(0x0020)], # ' '
            [chr(0x000A)], # '\n'
            [chr(0x000D)], # '\r'
            [chr(0x0009)]] # '\t'
    }

print(json.dumps(json_grammar, indent=4))
