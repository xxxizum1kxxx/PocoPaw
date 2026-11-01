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
    burstpipe/merge_freq.cpp
    burstpipe/demosaic.cpp
    burstpipe/ultrahdr.cpp
    burstpipe/pool.cpp
    burstpipe/third_party/rtprocess/rcd.cc
    burstpipe/third_party/rtprocess/amaze.cc
    burstpipe/third_party/rtprocess/border.cc)

# Same flags the measured prototype used; debug builds are optimised too, a burst takes minutes at -O0.
target_compile_options(burstpipe PRIVATE -O3 -ffast-math -march=armv8.2-a+fp16+dotprod -Wall -Wno-unused-function
    -Wno-unknown-pragmas -Wno-unused-variable -Wno-deprecated-declarations -Wno-nan-infinity-disabled -Wno-absolute-value)
# libultrahdr + its libjpeg-turbo, prebuilt for arm64 by pipeline/tools/build_ultrahdr.sh
set(UHDR ${CMAKE_CURRENT_SOURCE_DIR}/burstpipe/third_party/libultrahdr/lib/android)
target_link_libraries(burstpipe PRIVATE ${UHDR}/libuhdr.a ${UHDR}/libjpeg.a jnigraphics log)
target_link_options(burstpipe PRIVATE -Wl,-z,max-page-size=16384)
