#!/usr/bin/env bash

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH="$FROMAGER_ENV/lib;$FROMAGER_ENV/lib64" \
    -DCMAKE_INSTALL_RPATH="$FROMAGER_ENV/lib;$FROMAGER_ENV/lib64"
cmake --build build -j
