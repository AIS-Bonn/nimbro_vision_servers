# nimbro_vision_servers

Simple application to serve various vision models via http.

### Serving Models

See the individual READMEs for each model under the [models directories](./models/). Each model server has a corresponding Docker bringup which downloads and caches any required weights.

### Getting started

Click through [usage.ipynb](./example/usage.ipynb) and the individual examples under the [models directories](./models/). 

If you would like to use the helper functions there, install this package as directed below.

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

### ROS2 Integration

For use on a robot we recommend using the ros2 integration in [nimbro_api](https://github.com/AIS-Bonn/nimbro_api).

### License

This project is released under the **MIT License** - see [`LICENSE`](./LICENSE).

The bringups for various models may **download and import code, model weights or other components
from other projects**. These remain under **their own licences**, which may differ from MIT.

### Citation

If you find this package useful, please cite

[https://arxiv.org/abs/2503.16538](https://arxiv.org/abs/2503.16538)

```bibtex
@article{paetzold25detector,
  author  = {Bastian P{\"a}tzold and Jan Nogga and Sven Behnke},
  title   = {Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking},
  journal = {arXiv preprint arXiv:2503.16538},
  year    = {2025}
}
