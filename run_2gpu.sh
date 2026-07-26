#!/usr/bin/env bash
# Run ONE seed across both GPUs of a Kaggle T4x2 session (or any 2-GPU box).
# Four models per GPU, in parallel, then merge the two halves into seed<N>.csv.
#
#   bash run_2gpu.sh <seed> [epochs]
#
# The two model groups are balanced by measured runtime (WIRE is the slow one, so
# it is paired with the three fastest). Both groups share one protocol via the
# pinned --batch / --res, so the halves are poolable.

seed="${1:?usage: bash run_2gpu.sh <seed> [epochs]}"
epochs="${2:-500}"
common="--mesh nefertiti.obj --epochs ${epochs} --seeds ${seed} --batch 16384 --res 256"

echo ">> seed ${seed}: launching 4 models on each GPU (~3.3h on T4x2)"

CUDA_VISIBLE_DEVICES=0 python train_3dv2.py ${common} \
    --models wire siren fr finer --out "seed${seed}_gpu0.csv" &
CUDA_VISIBLE_DEVICES=1 python train_3dv2.py ${common} \
    --models gauss mfn fourier incode --out "seed${seed}_gpu1.csv" &
wait

g0="seed${seed}_gpu0.csv"; g1="seed${seed}_gpu1.csv"
if [ ! -s "$g0" ] || [ ! -s "$g1" ]; then
    echo "[error] a GPU job produced no output; check the logs above." >&2
    exit 1
fi

merged="seed${seed}.csv"
head -1 "$g0" > "$merged"
tail -n +2 "$g0" >> "$merged"
tail -n +2 "$g1" >> "$merged"
rows=$(($(wc -l < "$merged") - 1))
echo ">> wrote ${merged} (${rows} rows)"
[ "$rows" -eq 8 ] || echo "[warn] expected 8 rows, got ${rows} — a model may have crashed."
