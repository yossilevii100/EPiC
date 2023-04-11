## Unaugmented Architectures

- DGCNN, CurveNet, RPC, PCT, GDANet
```shell
python main.py --model dgcnn --eval
python main.py --model curvenet --eval
python main.py --model rpc --eval
python main.py --model pct --eval
python main.py --model gdanet --eval
```

## WolfMix augmented Architectures

- DGCNN, CurveNet, RPC, PCT, GDANet
```shell
python main.py --model dgcnn --eval --use_wolfmix
python main.py --model curvenet --eval --use_wolfmix
python main.py --model rpc --eval --use_wolfmix
python main.py --model pct --eval --use_wolfmix
python main.py --model gdanet --eval --use_wolfmix
```