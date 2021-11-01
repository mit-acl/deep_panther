## Instructions

First, install Python 3.7 (needed if you are in Ubuntu 18.04) following [this](https://stackoverflow.com/questions/51279791/how-to-upgrade-python-version-to-3-7/51280444#51280444)

Then create a virtual environment:

```bash
sudo apt-get install python3.7-venv
cd ~/installations/venvs_python/
python3.7 -m venv ./my37
printf '\nalias activate_my37="source ~/installations/venvs_python/my37/bin/activate"' >> ~/.bashrc 
```

Then, go to your `ws/src`, and do:

```bash
activate_my37
git clone git@github.com:jtorde/panther_compression.git
git submodule update --init
cd imitation
pip install numpy Cython wheel seals
pip install -e .
```

Now you can test the imitation repo by doing `python examples/quickstart.py`

You can test this repo by doing `bash run.sh` 


TODOS: see modifications (from Andrea) in imitation repo