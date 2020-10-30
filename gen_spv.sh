#!/usr/bin/env bash
dirname='assets/gen/spv'
mkdir -p "$dirname"
(cd assets/shaders; for a in *.{frag,vert}; do glslangValidator "$a" -V -o ../../"$dirname"/"$a".spv; done;)
