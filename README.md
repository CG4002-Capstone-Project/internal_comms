# internal_comms

## Setup
```
pip install -r requirements.txt
```

## Run 
- Run to print statements
```
python3 main.py --beetle_id 2 --dancer_id 0 --verbose True
```
- Run with model 
```
python3 main.py --beetle_id 2 --dancer_id 0 --debug True --model_type dnn --model_path ./dnn_model.pth --scaler_path ./dnn_std_scaler.bin
python3 main.py --beetle_id 2 --dancer_id 0 --debug True --model_type svc --model_path ./svc_model.sav --scaler_path ./svc_std_scaler.bin
```
- Run with dashboard
```
python3 main.py --beetle_id 2 --dancer_id 0 --debug True --model_type dnn --model_path ./dnn_model.pth --scaler_path ./dnn_std_scaler.bin --production True --dashboard True
```
- Collect data 
```
python3 main.py --beetle_id 2 --dancer_id 0 --collect True --verbose True
```
- Run in production
```
python3 main.py --beetle_id 2 --dancer_id 0 --production True --ultra96 True --dashboard True 
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