CKPT_PATH=checkpoint
L1_PATH=l1_prune_model.pth.tar


if [ -d "$CKPT_PATH" ]; then
    echo "[INFO] Exist Checkpoint"
else
    mkdir "$CKPT_PATH"
fi


if [ -f "$L1_PATH" ]; then
    mv "$L1_PATH" "$CKPT_PATH"
    echo "[INFO] MOVE l1 prune model"
else
    echo "[INFO] NOT Exist l1 prune model"
fi
