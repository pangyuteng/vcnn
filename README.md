# cnn infrastructure

# directory description
    \src\vcnn: code
    \src\vcnn\utils: helper methods
    \src\vcnn\conf.py: vcnn related configuration
    \exp: experiments, messy code
    \bin: entry points, script for pulling data
    \data: folder for storing training and testing data, not to be committed?
    \conf\model: folder for pre trained weights is cnn is trained.



# misc notes

### setup and activate virtualenvs
http://docs.python-guide.org/en/latest/dev/virtualenvs/

    cd my_project_folder
    virtualenv venv
    source venv/bin/activate

## dependencies
# gpu drive (varies)

sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt-get update

sudo apt-get install nvidia-352 nvidia-prime

http://ubuntuhandbook.org/index.php/2015/06/install-nvidia-352-21-ubuntu-1404/
http://askubuntu.com/questions/712067/unable-to-install-nvidia-driver-352


### cuda: 

* install cuda


    wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
    
https://developer.nvidia.com/cuda-downloads
    
* check if `nvcc` (cuda toolkit) is installed, if `nvcc` not found, try to locate `nvcc`, and add cuda to system path.


    nvcc    
    find /usr/local -name nvcc
    export CUDA_HOME=/usr/local/cuda-7.5
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64

http://deeplearning.net/software/theano/library/config.html#libdoc-config

http://askubuntu.com/questions/673124/nvcc-v-fails-but-cuda-7-0-installed-and-nvcc-present

http://kawahara.ca/theano-how-to-get-the-gpu-to-work/

https://groups.google.com/forum/#!topic/theano-users/KD7AcDMajFo


### install dependencies:
	    
	pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
	pip install Lasagne==0.1
		# pip install https://github.com/Lasagne/Lasagne/archive/master.zip
    pip install path.py
    pip install seaborn
    pip install Cython
    pip install h5py
    pip install theano
    pip install hyperopt
    pip install pymongo
	
    git clone git@github.com:pangyuteng/voxnet.git
    cd voxnet
    pip install --editable .

ref.

theano

http://deeplearning.net/software/theano/install.html

lasagne

http://lasagne.readthedocs.org/en/latest/user/installation.html

voxnet

https://github.com/dimatura/voxnet
https://github.com/pangyuteng/voxnet



### setup theano:
locate theano configuration file `.theanorc`.

    cd ~
    ls -a

updated `.theanorc` content, `pico .theanorc`:
    
    [global]
    floatX = float32
    device = gpu

    [lib]
    cnmem = 0

    [cuda] 
    root=/usr/local/cuda-7.5


test if theano is running using gpu:

`touch touchtheano.py`

`python touchtheano.py`


    # save file as touchtheano.py
    from theano import function, config, shared, sandbox
    import theano.tensor as T
    import numpy
    import time

    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')

        
http://deeplearning.net/software/theano/tutorial/using_gpu.html#testing-theano-with-gpu

### install other dependencies:
    sudo apt-get install git
    
    sudo apt-get install libblas-dev
    sudo apt-get install liblapack-dev
    sudo apt-get -y install libncurses-dev
    sudo pip install --upgrade pip
    sudo apt-get install python-pip python-dev build-essential 
    
    sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip
    sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
    sudo apt-get install python-scipy
    sudo pip install path.py
    sudo apt-get install libfreetype6-dev libpng-dev
    sudo pip install matplotlib    
    sudo pip install seaborn
    sudo pip install Cython
    sudo apt-get install libhdf5-dev
    sudo pip install h5py
    sudo pip install sklearn
    sudo pip install moviepy
    sudo pip install hyperopt
    sudo pip install pymongo
	
    git clone git@github.com:pangyuteng/voxnet.git
    cd voxnet
    sudo pip install --editable .
    
https://help.ubuntu.com/lts/serverguide/git.html

http://www.saltycrane.com/blog/2010/02/how-install-pip-ubuntu/

https://github.com/dimatura/voxnet

https://github.com/pangyuteng/voxnet

http://askubuntu.com/questions/350379/how-to-install-this-file-with-python

http://stackoverflow.com/questions/24744969/installing-h5py-on-an-ubuntu-server
http://stackoverflow.com/questions/21646179/how-to-install-python-matplotlib-in-ubuntu-12-04

### install vcnn

    git clone git@github.com:pangyuteng/vcnn.git

### configure and test vcnn

* edit email address in `config\email.yml`

* train and test simulated data, training report will be sent to configured email.


    cd \exp\simu
    python run.py

* keep process running and exit terminal
	
	tmux	
    command &
    disown
	tmux detach
	
	tmux list-sessions
	tmux attach -t 0
	
http://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session
http://unix.stackexchange.com/questions/89483/keeping-a-process-running-after-putty-or-terminal-has-been-closed

# todo:
create predict script
modify train/test/predict scripts to be a scikit like classes.


# useful links:

http://cs231n.github.io/
http://rodrigob.github.io/are_we_there_yet/build
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
https://github.com/ebenolson/pydata2015
http://keras.io/datasets/
https://github.com/sklearn-theano/sklearn-theano