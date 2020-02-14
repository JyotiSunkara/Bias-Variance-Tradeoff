all:
	python3 q1.py
	python3 q2.py

seeOne: 
	eog table1.png
	eog graph1.png

seeTwo:
	eog table2.png
	eog graph2.png

clean:
	rm -f table1.png table2.png graph1.png graph2.png ||: