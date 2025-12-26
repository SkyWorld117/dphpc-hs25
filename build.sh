#!/usr/bin/env bash

export HIGHS_DIR="/scratch/zyi/MyPackages/HiGHS"

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH="$FROMAGER_ENV/system/lib;$FROMAGER_ENV/system/lib64" \
    -DCMAKE_INSTALL_RPATH="$FROMAGER_ENV/system/lib;$FROMAGER_ENV/system/lib64"
cmake --build build --parallel
