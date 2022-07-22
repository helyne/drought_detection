# Data analysis
- Document here the project: drought_detection
- Description: A project for Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya
- Data Source: https://wandb.ai/wandb/droughtwatch/benchmark
- Type of analysis: Deep learning CNN multi-class image classification


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for drought_detection in gitlab.com/helyne.
If your project is not set please add it:

- Create a new project on `gitlab.com/helyne/drought_detection`
- Then populate it:

```bash
##   e.g. if group is "helyne" and project_name is "drought_detection"
git remote add origin git@github.com:helyne/drought_detection.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
drought_detection-run
```

# Install

Go to `https://github.com/helyne/drought_detection` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:helyne/drought_detection.git
cd drought_detection
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
drought_detection-run
```
