# internal_comms

## Setup
```
pip install -r requirements.txt
```

## Run 
- Run in debug mode
```
python myFinal50_3_beetles_edit2.py  --debug True --model_type cnn --model_path ./cnn_model.pth --scaler_path ./cnn_std_scaler.bin
```
- Run in production mode
```
python myFinal50_3_beetles_edit2.py 
```

## Linting
```
./fix_lint.sh
```