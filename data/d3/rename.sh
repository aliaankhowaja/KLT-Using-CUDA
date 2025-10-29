#!/bin/bash

# =================================================================
# Decrement PGM File Numbers
# Renames files from imgN.pgm to img(N-1).pgm.
# =================================================================

for old_name in $(ls -v img*.pgm 2>/dev/null); do
    if [[ "$old_name" =~ img([0-9]+)\.pgm ]]; then
        current_num="${BASH_REMATCH[1]}"
        new_num=$((current_num - 1))
        new_name="img${new_num}.pgm"
        echo "Renaming ${old_name} to ${new_name}"
        mv "$old_name" "$new_name"
    fi
done

echo "Renaming sequence complete."
