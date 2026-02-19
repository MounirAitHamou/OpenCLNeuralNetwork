set -e
set -o pipefail

cd "$(dirname "$0")"

bash setup.sh

bash runtests.sh
