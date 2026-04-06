if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <Tensor Name> <Binary Name> <Version> <Mode> <Counter File>"
    echo "Example: $0 Darpa ./correctness_in_progress 1 basic.yaml"
    exit 1
fi

NAME=$1
BINARY_NAME=$2
VERSION=$3
MODE=$4
COUNTER_FILE=$5

if [ ! -f "$COUNTER_FILE" ] && [ -f "yaml_files/$COUNTER_FILE" ]; then
    COUNTER_FILE="yaml_files/$COUNTER_FILE"
elif [ ! -f "$COUNTER_FILE" ] && [ -f "txt_files/$COUNTER_FILE" ]; then
    COUNTER_FILE="txt_files/$COUNTER_FILE"
fi

if [ ! -f "$COUNTER_FILE" ]; then
    echo "Warning: Counter file $COUNTER_FILE not found. Profiler may not output CSVs."
fi

cd ../mttkrp_tests

if [ "$NAME" == "Darpa" ]; then
    rocprofv3 -i  "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/darpa.bin "$MODE" 28436032 22476 22476 23776223 int
elif [ "$NAME" == "Freebase Music" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/fb-m.bin "$MODE" 99546550 23344784 23344784 166 int
elif [ "$NAME" == "Nell 1" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/nell-1.bin "$MODE" 143599552 2902330 2143368 25495389 int
elif [ "$NAME" == "Nell 2" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/nell-2.bin "$MODE" 76879419 12092 9184 28818 int
elif [ "$NAME" == "Chicago Crime" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/chicago_crime.bin "$MODE" 5330673 6186 24 77 32 int
elif [ "$NAME" == "Delicious" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/delicious.bin "$MODE" 140126220 532924 17262471 2480308 1443 int
elif [ "$NAME" == "Enron Emails" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/enron.bin "$MODE" 54202099 6066 5699 244268 1176 int
elif [ "$NAME" == "Flickr" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/flickr.bin "$MODE" 112890310 319686 28153045 1607191 731 int
elif [ "$NAME" == "NIPS" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/nips.bin "$MODE" 3101609 2482 2862 14036 17 int
elif [ "$NAME" == "Uber Pickups" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/uber_pickups.bin "$MODE" 3309490 183 24 1140 1717 int
elif [ "$NAME" == "VAST" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/vast.bin "$MODE" 26021945 165427 11374 2 100 89 int
elif [ "$NAME" == "LBNL Network" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/lbnl.bin "$MODE" 1698825 1605 4198 1631 4209 868131 int
elif [ "$NAME" == "LANL" ]; then
    rocprofv3 -i "../rocprof_tests/$COUNTER_FILE" -- ./"$BINARY_NAME" $VERSION ../../tensors/lanl.bin "$MODE" 69082467 3761 11154 8711 75147 9 int
else
    echo "Unknown Tensor: $NAME"
    exit 1
fi

cd ../rocprof_tests
mv ../mttkrp_tests/.rocprofv3/ .
mv ../mttkrp_tests/pass_* .
