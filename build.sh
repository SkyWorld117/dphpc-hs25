#!/usr/bin/env bash

export CUDSS_DIR=/scratch/zyi/MyPackages/libcudss-0.7.1.4
export TBB_DIR=/scratch/zyi/MyPackages/tbb-2022.3.0
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
