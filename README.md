<!---:lai-name: LightningDeepchecks--->

<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lai.png" width="250px">

⚡ Deepchecks App with [Lightning](https://lightning.ai) ⚡

[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai)
![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

______________________________________________________________________

</div>

## About

...

## Running the app

```bash
conda create --name lightning_deepchecks python=3.8
conda activate lightning_deepchecks

git clone https://github.com/Lightning-AI/LAI-Deepchecks-App
cd LAI-Deepchecks-App
pip install -e .

## To run the app locally
python -m lightning run app app.py

## To run the app on the cloud to share it with your peers and users
python -m lightning run app app.py --cloud
```

## Sneak Peek

![Deepchecks App](./visuals/home.png)
