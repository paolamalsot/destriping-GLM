#!/bin/bash
source ~/.bashrc

#echo "The directory of the sourced script is: $DIR"
make_outfile() {
    # Get the directory where this file is located
    local DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd $DIR/..
    root="$(pwd)"
    mkdir -p outfiles
    now=$(date +"%Y-%m-%d_%H-%M-%S")
    filename="$(basename $0 .sh)_$now.out"
    local outfile=outfiles/$filename

    cd $DIR
    echo "$outfile"
}