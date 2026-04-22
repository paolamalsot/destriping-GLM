source ~/.bashrc

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $DIR/../../utilities/are_we_in_cluster.sh
source $DIR/set_outfile.sh

# Use the is_cluster function
if is_cluster; then
    echo "we are in cluster"
    source $DIR/../../utilities/activate_py_env_biomed.sh
    activate_py_env_biomed
else
    source $DIR/../../utilities/activate_py_env.sh
    activate_py_env
fi

python_launch() {
    script_name="$1"
    python $DIR/python_launch.py --script_path $script_name
}