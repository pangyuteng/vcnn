# cnn infrastructure

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

### cuda: 

* check if `nvcc` (cuda toolkit) is installed, if `nvcc` not found, try to locate `nvcc`, and add cuda to system path.


    nvcc    
    find /usr/local -name nvcc
    export CUDA_HOME=/usr/local/cuda-7.5
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64

http://deeplearning.net/software/theano/library/config.html#libdoc-config

http://askubuntu.com/questions/673124/nvcc-v-fails-but-cuda-7-0-installed-and-nvcc-present

http://kawahara.ca/theano-how-to-get-the-gpu-to-work/

https://groups.google.com/forum/#!topic/theano-users/KD7AcDMajFo

### theano:
locate theano configuration file `.theanorc`.

    cd ~
    ls -a

updated `.theanorc` content, `pico .theanorc`:
    
    [global]
    floatX = float32
    device = gpu

    [lib]
    cnmem = 0.5

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

    pip install https://github.com/Lasagne/Lasagne/archive/master.zip
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
    
https://github.com/dimatura/voxnet

https://github.com/pangyuteng/voxnet


### install vcnn

    git clone git@gitlab.cvib.ucla.edu:pangyuteng/vcnn.git

### configure and test vcnn

* edit email address in `config\email.yml`

* train and test simulated data, training report will be sent to configured email.


    cd exp\1_simu
    python run.py


# todo:
create predict script
modify train/test/predict script to be a scikit like class?