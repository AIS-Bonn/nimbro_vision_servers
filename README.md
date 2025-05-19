# nimbro_vision_servers

Simple application to serve various vision models via http.

### Serving Models

See the individual READMEs for each model under the models/ directory. Each model server has a corresponding Docker bringup which downloads and caches any required weights.

### Getting started

Click through example/usage.ipynb and the individual examples under the models/ directory. 

If you would like to use the visualization functions there, install this package as directed below.

### Setup

```
git clone https://github.com/AIS-Bonn/nimbro_vision_servers.git
```

```
cd nimbro_vision_servers
```

```
conda create -y -n nimbro_vision_servers python=3.10
```

```
conda activate nimbro_vision_servers
```

```
pip install jupyter
```

```
pip install -e .
```

### License

This project is released under the **MIT License** - see [`LICENSE`](./LICENSE).

The bringups for various models may **download and import code, model weights or other components
from other projects**. These remain under **their own licences**, which may differ from MIT. 