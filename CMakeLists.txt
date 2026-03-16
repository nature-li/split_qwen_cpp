cmake_minimum_required(VERSION 3.18)
project(split_qwen_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(TRTLLM_LIBS /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs)

include_directories(
    /app/tensorrt_llm/include
    /usr/local/cuda/targets/x86_64-linux/include
    /opt/hpcx/ompi/include
)

link_directories(
    ${TRTLLM_LIBS}
    /usr/local/tensorrt/targets/x86_64-linux-gnu/lib
    /opt/hpcx/ompi/lib
)

set(TRTLLM_LINK_LIBS
    tensorrt_llm
    nvinfer_plugin_tensorrt_llm
    nvinfer
    pthread
    mpi
)

add_executable(infer src/infer.cpp)
target_link_libraries(infer ${TRTLLM_LINK_LIBS})