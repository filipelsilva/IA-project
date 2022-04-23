#!/bin/sh
trap ctrl_c INT

function ctrl_c() {
	echo "Aborted."
	rm tmp
	exit
}

for i in {0..10}; do
	echo "tests/input${i}.txt"
	time python numbrix.py tests/input${i}.txt > tmp
	cat tmp
	diff tmp tests/output${i}.txt
	echo ""
done

rm tmp