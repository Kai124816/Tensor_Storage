#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Tensor Name> <Mode>"
    echo "Example: $0 Darpa 1"
    exit 1
fi

NAME=$1
MODE=$2
CSV1=pass_1/out_counter_collection.csv
CSV2=pass_2/out_counter_collection.csv
OUTPUTCSV=tensor_performance_metrics.csv


# Exit immediately on error
set -e

# echo "Transfering Results to CSV"

python3 rocprof_pivot.py $CSV1 $CSV2 $OUTPUTCSV $NAME $MODE

echo "Deleting directories generated"
rm -r .rocprofv3/
rm -r pass_1
rm -r pass_2



