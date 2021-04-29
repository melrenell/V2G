# V2G-Sim-beta

## How to install
    1) Make sure you have Python 2.7 with Anaconda
    2) Download and install Gurobi Optimizer
    3) Download V2G-Sim
    4) Go inside the root folder of V2G-Sim and launch:

```shell
pip install -r requirement.txt
pip install .
```

That's it, you are all set up.
To use:
Create a driving itinerary and load profile similar to the examples in V2G/templates
Edit parameters in centralized_optimization.py
Run
A csv file will be generated with the outputs which create the plots
