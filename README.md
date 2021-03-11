# internal_comms

## Setup
```
pip install -r requirements.txt
```

## Run 
- Run in with model in debug mode
```
python3 main.py  --debug True --model_type dnn --model_path ./dnn_model.pth --scaler_path ./dnn_std_scaler.bin
python3 main.py  --debug True --model_type svc --model_path ./svc_model.sav --scaler_path ./svc_std_scaler.bin
```
- Collect data in train model
```
python main.py --train True
```
- Run in production mode
```
python main.py 
```

## Linting
```
./fix_lint.sh
```

## Docker
```
docker build -t ws .
nvidia-docker run --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 --net=host --privileged -v ~/internal_comms:/workspace -it ws bash
```