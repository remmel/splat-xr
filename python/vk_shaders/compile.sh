#!/bin/bash

compile_if_modified() {
    local src="$1"
    local spv="$2"

    # Check if SPV doesn't exist or source shader is newer
    if [ ! -f "$spv" ] || [ "$src" -nt "$spv" ]; then
        echo "Compiling $src..."
        glslangValidator -V "$src" -o "$spv"
    fi
}

compile_if_modified splats.vert splats_vert.spv
compile_if_modified splats.geom splats_geom.spv
compile_if_modified splats.frag splats_frag.spv

compile_if_modified points.vert points_vert.spv
compile_if_modified points.geom points_geom.spv
compile_if_modified points.frag points_frag.spv