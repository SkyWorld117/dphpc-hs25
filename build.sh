#!/usr/bin/env bash

export CUDSS_DIR=/scratch/zyi/MyPackages/libcudss-0.7.1.4
cmake -B build
cmake --build build -j