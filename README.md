
# Reinforcement Learning Lab

Created in Cython for Robotic Agents (J-Term 2022) at Pacific Lutheran University. Given a 10x15 grid containing a door, 
assuming the door rewards 100, and each other space rewards -1, constructs a Q table.



## Authors

- Kieran Kim-Murphy ([@kjkm](https://github.com/kjkm))

## Requirements
To install requirements, navigate to source and run

```
pip install -r requirements.txt
```

## Build

To build this project, navigate to source and run

```bash
  python setup.py build_ext --inplace
```




## In PyCharm

To build this project in PyCharm, open the project folder. Then, click `Tools \ Run setup.py Task`. In the window that pops up, select `build_ext`. Then enter `--inplace` in the command line window that appears. Click `OK`, and the `q_learning` module should build successfully.

## Run

To run this program, run

```
python rl.py 
```


## Configurations
To configure this program, edit constant parameter in `rl.py`. 