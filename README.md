# Reproducer

Reproducer is an image reproduction project with genetic algortihm. It's extended and improved implementation (more accurate and faster reproduction for all colors) of <a href="https://github.com/ahmedfgad/GARI">ahmetdfgad/GARI</a> project.

## Installation

- Clone the repository.
- `$ cd reproducer && virtualenv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- Specify an image path in the code (which is in main function)
- Specify width and height multiplier in main function if you want to resize picture or have faster iterations.
- Specify the iteration count, default is `10m`.
- `$ python3 reproducer.py`
- You can find outputs in `points/` folder.

## Strategies

- If you lower your picture resolution, you'll have more accurate reproduced images. Also `reproducer` will work faster.
- If you want to have faster reproduction and also same resolution, you can use upscalers like <a href="https://github.com/idealo/image-super-resolution">image-super-resolution</a>.
- You can reproduce your friends' photos and let them use their reproduced photos as profile photo! That's funny, strongly recommended.
- Mostly, below 400x400 images are over 90% percentage being reproduced well in `5m` iterations.

## Example (Thanks to <b>Elon Musk</b>)

<div style="display: flex;">
  <div style="display: flex; text-align: center; flex-direction: column;">
    <span>Iterations: 0</span>
    <img src='assets/solution_0.png' />
  </div>
  &nbsp;
  <div style="display: flex; text-align: center; flex-direction: column;">
    <span>Iterations: 5k</span>
    <img src='assets/solution_5000.png' />
  </div>
  &nbsp;
  <div style="display: flex; text-align: center; flex-direction: column;">
    <span>Iterations: 10k</span>
    <img src='assets/solution_10000.png' />
  </div>
  &nbsp;
  <div style="display: flex; text-align: center; flex-direction: column;">
    <span>Iterations: 50k</span>
    <img src='assets/solution_50000.png' />
  </div>
</div>

<div style="display: flex;">
  <div style="display: flex; text-align: center; flex-direction: column;">
    <span>Iterations: 100k</span>
    <img src='assets/solution_100000.png' />
  </div>
  &nbsp;
  <div style="display: flex; text-align: center; flex-direction: column;">
    <span>Iterations: 250k</span>
    <img src='assets/solution_250000.png' />
  </div>
  &nbsp;
  <div style="display: flex; text-align: center; flex-direction: column;">
    <span>Iterations: 500k</span>
    <img src='assets/solution_500000.png' />
  </div>
  &nbsp;
  <div style="display: flex; text-align: center; flex-direction: column;">
    <span>Iterations: 1m</span>
    <img src='assets/solution_1000000.png' />
  </div>
</div>