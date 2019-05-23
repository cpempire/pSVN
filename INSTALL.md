                        Inverse Problem PYthon library

```
 __        ______  _______   _______   __      __  __  __  __       
/  |      /      |/       \ /       \ /  \    /  |/  |/  |/  |      
$$ |____  $$$$$$/ $$$$$$$  |$$$$$$$  |$$  \  /$$/ $$ |$$/ $$ |____  
$$      \   $$ |  $$ |__$$ |$$ |__$$ | $$  \/$$/  $$ |/  |$$      \ 
$$$$$$$  |  $$ |  $$    $$/ $$    $$/   $$  $$/   $$ |$$ |$$$$$$$  |
$$ |  $$ |  $$ |  $$$$$$$/  $$$$$$$/     $$$$/    $$ |$$ |$$ |  $$ |
$$ |  $$ | _$$ |_ $$ |      $$ |          $$ |    $$ |$$ |$$ |__$$ |
$$ |  $$ |/ $$   |$$ |      $$ |          $$ |    $$ |$$ |$$    $$/
$$/   $$/ $$$$$$/ $$/       $$/           $$/     $$/ $$/ $$$$$$$/
```


                          https://hippylib.github.io

`hIPPYlib` depends on [FEniCS](http://fenicsproject.org/) versions 
2016.1, 2016.2, 2017.1, 2017.2.  
The suggested version of `FEniCS` to use with `hIPPYlib` is
2017.2.
**Note**: `FEniCS 2018.1` is not
supported by `hIPPYlib`.

`FEniCS` needs to be built with the following dependecies enabled:

 - `numpy`, `scipy`, `matplotlib`, `mpi4py`
 - `PETSc` and `petsc4py` (version 3.7.0 or above)
 - `SLEPc` and `slepc4py` (version 3.7.0 or above)
 - PETSc dependencies: `parmetis`, `scotch`, `suitesparse`, `superlu_dist`, `ml`, `hypre`
 - (optional): `mshr`, `jupyter`
 

## Run FEniCS from Docker (Linux, MacOS, Windows)

An easy way to run `FEniCS` is to use their prebuilt `Docker` images.

First you will need to install [Docker](https://www.docker.com/) on
your system.  MacOS and Windows users should preferably use `Docker
for Mac` or `Docker for Windows` --- if it is compatible with their
system --- instead of the legacy version `Docker Toolbox`.

Among the many docker's workflow discussed [here](http://fenics.readthedocs.io/projects/containers/en/latest/quickstart.html),
we suggest using the `Jupyter notebook`[one](http://fenics.readthedocs.io/projects/containers/en/latest/jupyter.html).

### Docker for Mac, Docker for Windows and Linux users (Setup and first use instructions)

We first create a new Docker container to run the `jupyter-notebook`
command and to expose port `8888`.  From a command line shell, go to
the `hippylib` folder and type:
```
docker run --name hippylib-nb -w /home/fenics/hippylib -v $(pwd):/home/fenics/hippylib -d -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable:2017.2.0 'jupyter-notebook --ip=0.0.0.0'
docker logs hippylib-nb
```
The notebook will be available at
`http://localhost:8888/?token=<security_token_for_first_time_connection>`
in your web browser.  From there you can run the interactive notebooks
or create a new shell (directly from your browser) to run python
scripts.

### Docker Toolbox users on Mac/Windows (Setup and first use instructions)

`Docker Toolbox` is for older Mac and Windows systems that do not meet
the requirements of `Docker for Mac` or `Docker for Windows`.  `Docker
Toolbox` will first create a lightweight linux virtual machine on your
system and run docker from the virtual machine.  This has implications
on the workflow presented above.

We first create a new `Docker` container to run the `jupyter-notebook` command and to expose port `8888` on the virtual machine.
From a command line shell, go to the `hippylib` folder and type:
```
docker run --name hippylib-nb -w /home/fenics/hippylib -v $(pwd):/home/fenics/hippylib -d -p $(docker-machine ip $(docker-machine active)):8888:8888 quay.io/fenicsproject/stable:2017.2.0 'jupyter-notebook --ip=0.0.0.0'
docker logs hippylib-nb
```
To find out the IP of the virtual machine we type:
```
docker-machine ip $(docker-machine active)
```

The notebook will be available at `http://<ip-of-virtual-machine>:8888/?token=<security_token_for_first_time_connection>` in your web browser.
From there you can run the interactive notebooks or create a new shell (directly from your browser) to run python scripts.

### Subsequent uses
The docker container will continue to run in the background until we stop it:
```
docker stop hippylib-nb
```
To start it again just run:
```
docker start hippylib-nb
```
If you would like to see the log output from the Jupyter notebook server (e.g. if you need the security token) type:
```
docker logs hippylib-nb
```

## Install FEniCS from Conda (Linux or MacOS)

To use the prebuilt Anaconda Python packages (Linux and Mac only),
first install [Anaconda3](https://docs.continuum.io/anaconda/install),
then run following commands in your terminal:

```    
conda create -n fenicsproject -c conda-forge fenics==2017.2.0
source activate fenicsproject
conda install mpi4py
conda install matplotlib
conda install scipy
conda install sympy==1.1.1
conda install jupyter
```
> **Note**: FEniCS Anaconda recipes are maintained by the FEniCS community and distributed binary packages do not have a full feature set yet, especially regarding sparse direct solvers and input/output facilities.

## Build FEniCS from source using hashdist (Linux and MacOS 10.12 or below)

To build `FEniCS` from source we suggest using the scripts and profile
files in `fenics-hashdist`. These scripts and profile files contain
small modifications with respect to the ones provided by the `FEniCS`
community to ensure that all the dependencies needed by `hIPPYlib` are
installed.

See `fenics-hashdist/README.md` for further details.


## Other ways to build FEniCS

For instructions on other ways to build `FEniCS`,
we refer to the FEniCS project [download
page](https://fenicsproject.org/download/).  Note that this
instructions always refer to the latest version of `FEniCS` which may or
may not be yet supported by `hIPPYlib`. Always check the `hIPPYlib`
website for supported `FEniCS` versions.

## Build the hIPPYlib documentation using Sphinx

To build the documentation you need to have `sphinx` (tested on v.1.7.5),
`m2r` and `sphinx_rtd_theme` - all of these can be installed via `pip`.

To build simply run `make html` from `doc` folder.



