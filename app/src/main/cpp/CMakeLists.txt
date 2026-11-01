cmake_minimum_required(VERSION 3.22.1)
project(burstpipe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(burstpipe SHARED
    jni_burst.cpp
    burstpipe/burst.cpp
    burstpipe/align.cpp
    burstpipe/merge.cpp
    burstpipe/kr.cpp
    burstpipe/finish.cpp
    burstpipe/pool.cpp)

# Same flags the measured prototype used; debug builds are optimised too, a burst takes minutes at -O0.
target_compile_options(burstpipe PRIVATE -O3 -ffast-math -march=armv8.2-a+fp16+dotprod -Wall -Wno-unused-function)
target_link_libraries(burstpipe PRIVATE jnigraphics log)
target_link_options(burstpipe PRIVATE -Wl,-z,max-page-size=16384)
