#!/bin/bash

compile_if_modified() {
    local src="$1"
    local spv="$2"
    local stage="$3"
    local STAGE="${stage^^}"

    # Check if SPV doesn't exist or source shader is newer
    if [ ! -f "$spv" ] || [ "$src" -nt "$spv" ]; then
        echo "Compiling $src..."
        glslangValidator -V -g "$src" -o "$spv" -S "$stage" -D"$STAGE"
    fi
}

compile_if_modified splats.glsl spv/splats_vert.spv vert
compile_if_modified splats.glsl spv/splats_geom.spv geom
compile_if_modified splats.glsl spv/splats_frag.spv frag
