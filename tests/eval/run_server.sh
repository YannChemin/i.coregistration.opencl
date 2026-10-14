#!/usr/bin/env bash
# Run an evaluation of i.coregistration.opencl on the GPU server.
#
# Usage:
#   tests/eval/run_server.sh <tag> synthetic <reference.tif> [eval_synthetic.py options]
#   tests/eval/run_server.sh <tag> basemaps <a.tif> <b.tif> [c.tif ...] [-- eval_basemaps.py options]
#
# Builds the working tree on the server into ~/build/i.coregistration.opencl_<tag>
# (its own bin/, docs and man folders: the installed module is left alone),
# uploads the images to a private /tmp/coreg_eval_<tag> (mode 700: the data
# may be confidential), runs the evaluation detached in a throwaway GRASS
# project, polls it, fetches the results into tests/eval/output/<tag>/ and
# deletes the remote folder. Progress goes to MetaMonitor
# (https://github.com/YannChemin/MetaMonitor) as slot coreg-eval-<tag>.
#
# Settings through the environment: REMOTE (default yann@10.42.0.89),
# RUSTICL (default radeonsi), BUILD=0 to reuse the server build,
# BUILD_TAG to use the build of another tag.

set -euo pipefail

[ $# -ge 3 ] || { sed -n '2,8p' "$0"; exit 1; }
TAG=$1; KIND=$2; shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
REMOTE="${REMOTE:-yann@10.42.0.89}"
RUSTICL="${RUSTICL:-radeonsi}"
BUILD="${BUILD:-1}"
RBUILD="build/i.coregistration.opencl_${BUILD_TAG:-$TAG}"
WORK="/tmp/coreg_eval_$TAG"
OUT="$HERE/output/$TAG"
SLOT="coreg-eval-$TAG"

ssh_r() { ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" "$@"; }
mm() { command -v mmctl >/dev/null && mmctl "$@" >/dev/null 2>&1 || true; }

IMAGES=()
EXTRA=()
case "$KIND" in
    synthetic) IMAGES=("$1"); shift; EXTRA=("$@") ;;
    basemaps)
        while [ $# -gt 0 ] && [ "$1" != "--" ]; do IMAGES+=("$1"); shift; done
        [ $# -gt 0 ] && shift
        EXTRA=("$@") ;;
    *) echo "Unknown evaluation <$KIND>" >&2; exit 1 ;;
esac

trap 'mm fail "$SLOT" -d "stopped, see $OUT/eval.log"' ERR
mm set "$SLOT" 0 -n "i.coregistration.opencl eval $TAG" -d "build" --pid $$

ssh_r true || { echo "Remote $REMOTE unreachable." >&2; exit 1; }
if [ "$BUILD" = 1 ]; then
    echo "== Build on $REMOTE:$RBUILD"
    ssh_r "mkdir -p $RBUILD"
    rsync -a --delete --exclude 'OBJ.*' --exclude '__pycache__' --exclude '.git' \
        --exclude references --exclude .pytest_cache --exclude tests/eval/output \
        "$SRC/" "$REMOTE:$RBUILD/"
    ssh_r "cd $RBUILD && make MODULE_TOPDIR=/usr/local/grass86 OCLLIB=-lOpenCL \
        BIN=\$HOME/$RBUILD/bin HTMLDIR=\$HOME/$RBUILD/html MDDIR=\$HOME/$RBUILD/mkdocs \
        MANBASEDIR=\$HOME/$RBUILD/man > build.log 2>&1 || (tail -20 build.log; exit 1)"
fi

echo "== Upload to $REMOTE:$WORK"
ssh_r "rm -rf $WORK && mkdir -p -m 700 $WORK/out"
REMOTE_IMAGES=()
for f in "${IMAGES[@]}"; do
    rsync -a "$f" "$REMOTE:$WORK/"
    REMOTE_IMAGES+=("$WORK/$(basename "$f")")
done
rsync -a "$HERE/eval_$KIND.py" "$REMOTE:$WORK/"

# Extra arguments quoted for the remote shell.
EXTRA_Q=""
[ ${#EXTRA[@]} -gt 0 ] && EXTRA_Q=$(printf '%q ' "${EXTRA[@]}")
ssh_r "cat > $WORK/run.sh" <<EOS
set -e
cd $WORK
grass --tmp-project ${REMOTE_IMAGES[0]} --exec python3 $WORK/eval_$KIND.py ${REMOTE_IMAGES[*]} \
    --output $WORK/out --module \$HOME/$RBUILD/bin/i.coregistration.opencl --tag $TAG $EXTRA_Q
EOS
ssh -f -n -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" \
    "cd $WORK && setsid nohup bash -c 'RUSTICL_ENABLE=$RUSTICL bash run.sh > eval.log 2>&1 && touch done || touch failed' >/dev/null 2>&1 &"

echo "== Running ($KIND)"
while ! ssh_r "test -e $WORK/done -o -e $WORK/failed"; do
    step=$(ssh_r "grep -E '^[0-9]+/[0-9]+ ' $WORK/eval.log 2>/dev/null | tail -1" || true)
    if [ -n "$step" ]; then
        k=${step%%/*}; rest=${step#*/}; n=${rest%% *}
        mm set "$SLOT" "$k" --of "$n" -n "i.coregistration.opencl eval $TAG" -d "$KIND: ${rest#* }" --pid $$
    fi
    sleep 15
done

mkdir -p "$OUT"
rsync -a "$REMOTE:$WORK/eval.log" "$REMOTE:$WORK/out/" "$OUT/"
FAILED=0
ssh_r "test -e $WORK/failed" && FAILED=1
# The images may be confidential: the remote folder goes in any case.
ssh_r "rm -rf $WORK"
ssh_r "ls -d /tmp/grass8-* 2>/dev/null | while read d; do p=\${d##*-}; kill -0 \$p 2>/dev/null || rm -rf \$d; done" || true
if [ "$FAILED" = 1 ]; then
    tail -30 "$OUT/eval.log"
    echo "Evaluation failed, log in $OUT/eval.log." >&2
    false
fi
mm done "$SLOT" -d "$KIND results in tests/eval/output/$TAG"
cat "$OUT"/*.md
