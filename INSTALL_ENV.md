# Install environment

install `python3` and virtualenv
set `python3` as default: Open your `.bashrc` file `vim ~/.bashrc`. Type `alias python=python3`

```sh
git clone https://github.com/lukinma/dlcourse_ai.git
cd dlcourse_ai
virtualenv .venv -p python3
.venv/bin/pip install jupyter notebook tornado\<6
.venv/bin/pip install -r assignments/assignment1/requirements.txt
```
>Note: our environment does not work with tornado 6, so we use oldier tornado.

To start jupyter notebook with assignment #1 run command:
```sh
.venv/bin/jupyter notebook assignments/assignment1/
```