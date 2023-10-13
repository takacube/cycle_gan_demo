## Cycle Gan Image Style Changer

- set up
```bash
docker compose up -d
docker exec -it cycle_gan bash
pip3 install -r requirments.txt
```

- experiment
```bash
python3 main.py
```



## PipeLine

developed pipeline with Luigi
```
- DataLoader
- PreProcess
- Training
- Sample
```
