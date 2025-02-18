#!/bin/bash

compile_if_modified() {
    local src="$1"
    local spv="$2"

    # Check if SPV doesn't exist or source shader is newer
    if [ ! -f "$spv" ] || [ "$src" -nt "$spv" ]; then
        echo "Compiling $src..."
        glslangValidator -V -g "$src" -o "$spv"
    fi
}

compile_if_modified splats.vert spv/splats_vert.spv
compile_if_modified splats.geom spv/splats_geom.spv
compile_if_modified splats.frag spv/splats_frag.spv
