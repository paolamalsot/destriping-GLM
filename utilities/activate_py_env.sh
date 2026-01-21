source ~/.bashrc

activate_py_env(){
    # Get the directory where file_y.sh is located
    local DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    cd $DIR/..

    if command -v pixi >/dev/null 2>&1; then
        # Activate pixi environment WITHOUT launching a subshell
        eval "$(pixi shell-hook)"
    fi

    export PYTHONPATH="${PYTHONPATH}:${pwd}"
}