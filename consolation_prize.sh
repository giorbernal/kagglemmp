#!/bin/bash

if [ $# != 1 ]; then
	echo 'Error: introduce a numeric unique param.'
	exit -1
fi

echo "Init consolation prize process ..."

PARTS=$1

for (( i=1; i <= $PARTS; ++i ))
do
	echo "---------- Init process Part $i ----------"
	python -u mmp_STEP_Consolation_Prize.py $i
	echo "---------- End  process Part $i ----------"
done

echo "Consolation prize process finished"
