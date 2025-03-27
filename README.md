# TIG automated innovator prototype

LLM-powered automated innovator for TIG ([document](https://www.overleaf.com/project/676408860ebce2fc210a5309)).

See `notebooks/` folder for research notes and demonstrations.
See `scripts/` folder for running large scale experiments. 


<details>
<summary><b>Instructions to set up local Python environment 💻 </b></summary>

To run the notebooks, we need to install software dependencies in Python 3 (3.7 or higher).
    
1. First, open the terminal and create a Python 3 virtual environment in the repository directory

```
mkdir venv/
python3 -m venv venv/
```

Note it is a good habit to specify the exact Python version `python3.xx -m venv venv/`
    
2. Now activate it 
    
```
. venv/bin/activate
```
    
3. Finally, install the required dependencies 
    
```
python3 -m pip install -r requirements.txt
```

4. Set up the `.env` file in the root directory for local paths and variables

```
VENV_PATH="/your/venv/path/..."
CHECKPOINT_PATH="/your/checkpoint/path/..."
```

   
</details>


<details>
<summary><b>Instructions to set up IPython kernel for Jupyter notebooks</b></summary>


1. Add the new environment to Jupyter kernels 

```
python3 -m ipykernel install --user --name=TIG_grad
```
  
2. Now one should be able to run the notebooks with all dependencies available using the `TIG_grad` IPython kernel.

</details>