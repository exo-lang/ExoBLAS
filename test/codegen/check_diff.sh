target_arch=$1
level=$2
kernel=$3

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_DIR=$SCRIPT_DIR/../../build/$target_arch

GENERATED_KERNEL_DIR=$BUILD_DIR/src/$level/$kernel.exo/
EXPECTED_DIR=$SCRIPT_DIR/expected

HEADER_NAME=$kernel.h
SOURCE_NAME=$kernel.c

GENERATED_HEADER_PATH=$GENERATED_KERNEL_DIR/$HEADER_NAME
GENERATED_SOURCE_PATH=$GENERATED_KERNEL_DIR/$SOURCE_NAME

EXPECTED_HEADER_PATH=$GENERATED_KERNEL_DIR/$HEADER_NAME
EXPECTED_SOURCE_PATH=$GENERATED_KERNEL_DIR/$SOURCE_NAME

diff $GENERATED_HEADER_PATH $EXPECTED_HEADER_PATH

if [ $? -neq 0 ]; then
  echo "Error: checking the diff of the header files failed!"
  exit 1
fi

diff $GENERATED_SOURCE_PATH $EXPECTED_SOURCE_PATH

if [ $? -neq 0 ]; then
  echo "Error: checking the diff of the source files failed!"
  exit 1
fi
