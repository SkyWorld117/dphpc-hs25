#!/usr/bin/env bash

set -Eeuo pipefail

export CUDSS_DIR="/scratch/zyi/MyPackages/libcudss-0.7.1.4"

cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_BUILD_RPATH="$FROMAGER_ENV/system/lib;$FROMAGER_ENV/system/lib64" \
    -DCMAKE_INSTALL_RPATH="$FROMAGER_ENV/system/lib;$FROMAGER_ENV/system/lib64" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel

# Generate .clangd file for clangd
echo "CompileFlags:
  Add:
    -  --cuda-path=$FROMAGER_ENV/system/
    -  --cuda-gpu-arch=sm_90
    -  -L$FROMAGER_ENV/system/lib64
    -  -I$FROMAGER_ENV/system/include
    -  -I$CUDSS_DIR/include
    -  -I$PWD/src
    -  -I$PWD/include
" > .clangd

python3 -c "
import json
import os

db_path = 'build/compile_commands.json'
if os.path.exists(db_path):
    with open(db_path, 'r') as f:
        data = json.load(f)
    
    target_file = '$PWD/src/dual_simplex/phase2_dual.cu'
    new_cmd = '$FROMAGER_ENV/system/bin/nvcc -O3 -DNDEBUG -lineinfo -std=c++17 -Werror=cross-execution-space-call -Wno-deprecated-declarations -Wno-error=non-template-friend -fopenmp -lineinfo -x cu -c $PWD/src/dual_simplex/phase2_dual.cu -o CMakeFiles/cuopt.dir/src/dual_simplex/phase2_dual.cu.o'
    
    for entry in data:
        if entry['file'] == target_file:
            entry['command'] = new_cmd
            
    with open(db_path, 'w') as f:
        json.dump(data, f, indent=2)
"

set +Eeuo pipefail