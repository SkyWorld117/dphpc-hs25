#!/usr/bin/env bash

export CUDSS_DIR="/scratch/zyi/MyPackages/libcudss-0.7.1.4"

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH="$FROMAGER_ENV/system/lib;$FROMAGER_ENV/system/lib64" \
    -DCMAKE_INSTALL_RPATH="$FROMAGER_ENV/system/lib;$FROMAGER_ENV/system/lib64" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel
