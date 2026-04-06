#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <Tensor Name> <Mode> <Counter .txt File> <Number of CSVs> <Output CSV>"
    echo "Example: $0 Darpa 1 basic.txt 2"
    exit 1
fi

NAME=$1
MODE=$2
COUNTERS=$3
NUM_CSVS=$4
OUTPUTCSV=$5
CSV1=pass_1/out_counter_collection.csv
CSV2=pass_2/out_counter_collection.csv
CSV3=pass_3/out_counter_collection.csv


# Exit immediately on error
set -e

# echo "Transfering Results to CSV"
if [ "$NUM_CSVS" -eq 1 ]; then
    python3 rocprof_pivot.py \
        --passes $CSV1 \
        --output $OUTPUTCSV \
        --tensor $NAME \
        --mode $MODE \
        --counters-file $COUNTERS
elif [ "$NUM_CSVS" -eq 2 ]; then
    python3 rocprof_pivot.py \
        --passes $CSV1 $CSV2 \
        --output $OUTPUTCSV \
        --tensor $NAME \
        --mode $MODE \
        --counters-file $COUNTERS
elif [ "$NUM_CSVS" -eq 3 ]; then
    python3 rocprof_pivot.py \
        --passes $CSV1 $CSV2 $CSV3 \
        --output $OUTPUTCSV \
        --tensor $NAME \
        --mode $MODE \
        --counters-file $COUNTERS
else 
    echo "You can pass in one, two, or three CSVs in this script. $NUM_CSVS were passed in."
    exit 1
fi

echo "Deleting directories generated"
rm -rf .rocprofv3/ pass_1 pass_2 pass_3
