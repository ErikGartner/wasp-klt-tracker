# Kanade-Lucas-Tomasi Tracker
*An implementation of the KTL tracker for the WASP course.*

The details of algorithm is describe in the [notebook](Solution.ipynb)

## How to run

Install requirements (optionally into a virtual env):
```bash
pip install -r requirements.txt
```

Run `python main.py -h` for help.
```
usage: main.py [-h] [--custom] [source]

KTL Tracker

positional arguments:
  source      The source, defaults to webcame, else filepath to a video.

optional arguments:
  -h, --help  show this help message and exit
  --custom    use my custom KTL implementation.
```

To run using the webcam:
```bash
python main.py
```

To run on a video:
```bash
python main.py data/coke_zero.mp4
```

## License
MIT
