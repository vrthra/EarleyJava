all: javaparser
	@echo done.

debug=-m pudb
pythonparser:
	python3 $(debug) src/Parser.py EarleyParser/grammar.json EarleyParser/myfile.txt

cp=~/.m2/repository/org/json/json/20160810/json-20160810.jar:EarleyParser/target/classes
javaparser:
	java -cp $(cp) parser.App EarleyParser/grammar.json EarleyParser/myfile.txt
