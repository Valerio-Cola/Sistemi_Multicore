# Sapienza (CS) Cluster

# Getting an account

Ask me to create an account. If you are a student of the *Multicore Programming* course, I’ll ask you to fill a form towards the end of the course.

In the following, I’ll indicate with `user` your username.

# Accessing the cluster

You need to login into the **submitter** node of the cluster (i.e., the node from which you can run your jobs/applications).

## If you are connected to Sapienza’s network

```bash
ssh -J user@151.100.174.45 user@submitter
```

## If you are not connected to Sapienza’s network

### Download and install the VPN

1. Download the package at the following link: https://drive.google.com/file/d/14G95lT9PExqIJ1xf942dh_1G22wlQYr1/view
2. Install the .exe file
3. Run the Open VPN program
4. Connect to the VPN using the credential that you normally use to access Sapienza’s services

### Access the cluster

```bash
ssh user@192.168.0.102
```

<aside>
⚠️

You need to open and connect to the VPN everytime you want to connect to the cluster (i.e., before SSHing, you need to do steps 3. and 4. from the “Download and install the VPN” section).

</aside>

# Setup the environment

You should add the following to your `.bashrc`

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

# Compile your application

You can compile your application from the submitter node. The compute capability of the GPUs of the cluster is `sm_75` (you can get it with `nvidia-smi --query-gpu=compute_cap --format=csv` but you must run it from a node that has a GPU — not from the submitter node (see below)). The GPUs of the cluster are `NVIDIA Quadro RTX 6000` GPUs (two per node, not all the nodes have GPUs).

# Run your application

The cluster uses HTCondor (sic! 😟) to access the compute nodes. This means that before using a specific node/server, you need to ask HTCondor to provide you access to it.

We will discuss separately how to get access to a single node, to run application running on a single node (e.g., a CUDA application using one GPU, or an OpenMP application), and then we will discuss how to run MPI application spanning over multiple nodes.

## Single-node applications

There are two ways of getting access to a compute node. An **interactive** way, where you will have access to a bash shell running on a given node, and a **batch** mode, where you will write a small script describing which commands must be run on that node, and which will be *enqueued* and executed at a later time (e.g., if the node you want to use is not yet available).

**Batch mode** is the suggested mode for GPU applications.  

**Interactive mode** can be used for single-node OpenMP applications.

### Batch mode

1. Prepare a `job.sub` file with the following content:
    
    ```bash
    universe = vanilla
    
    # Where should the output/error be saved
    log = cuda_job.log
    output = cuda_job.out
    error = cuda_job.err
    
    # Request GPU resources
    request_gpus = 1
    
    # Specify any environment setup if needed
    getenv = True
    
    queue
    ```
    
2. Submit the job (i.e., ask the scheduler to find a node with the requested resource (1 GPU), and to run the application on that node)
    
    ```bash
    condor_submit job.sub -append 'executable = /path/to/your_executable' -append 'arguments = arg1 arg2'
    ```
    
    <aside>
    ⚠️
    
    Replace `/path/to/your_executable` with your application name/path and `arg1 arg2` with the arguments of your application.  If you need to run more than one command (e.g., to prepare files before being processed by the applications), wrap everything into a bash script and use that as your executable.
    
    </aside>
    
3. You can check the status of your job with:
    
    ```bash
    condor_q
    ```
    
    This will tell you whether the job is waiting to be executed, is running, is on hold, etc…
    
4. Once the job terminates. You can find in the `cuda_job.*` files the content of whatever it print on standard output/error.
5. If you want to kill a job that is waiting or running
    
    ```bash
    condor_rm <jobId>
    ```
    
    Where `jobId` is the id that gets printed when you submit your job (you can retrieve it through `condor_q` )
    

### Interactive mode

1. Check which nodes are online: `condor_status` 
    1. To check how many GPUs a node has (e.g., for `node113`): `condor_status -long -const 'Machine == "node113.di.rm1"' | grep TotalGPUs` 
    2. To check how many CPUs a node has (e.g., for `node113`): `condor_status -long -const 'Machine == "node113.di.rm1"' | grep TotalCpus` 
    3. To check how much memory a node has (e.g., for `node113`): `condor_status -long -const 'Machine == "node113.di.rm1"' | grep TotalMemory`
2. To get interactive access to a specific node (e.g., to `node113`):  `condor_submit -interactive -append 'requirements = (Machine == "node113.di.rm1")'`
    1. To get interactive access to a node with a GPU (without specifying which node exactly): `condor_submit -interactive -append "requirements = (GPUs > 0)"`   
    2. **ATTENTION:** By default, you might share the node with other users (ok for development/debug but not ideal if you need to collect performance data)
    3. To get exclusive access to a node (i.e., be the only user using that node):
        - First, you need to get how much GPUs, CPUs, and memory that node has
        - Then, you can get access to it (e.g., to `node113`) with: `condor_submit -interactive -append "request_cpus = 64"  -append "request_memory = 257663"  -append "request_gpus = 2"  -append 'requirements = (Machine == "node113")'`
            - **ATTENTION:** You need to replace the correct number of CPUs, GPUs, etc… with the values collected in points 1.a, 1.b, 1.c
            - In this example, `node113` might be used by someone else. If you just want a node (to use exclusively), which has GPUs, replace the `requirements`  part with `requirements = (GPUs > 0)`
        - **ATTENTION:** If the node you want to get access to is used by someone else, you might have to wait until the node is released.
3. Do CTRL-D or `exit` once you finished using the node

## Multi-Node MPI Applications

For MPI applications we are only going to use the batch mode.

### Tips

- create a *logs* folder to save your application results

## Needed Files

- openmpiscript (provided by HTCondor)
- <you_app_name>.job
- <you_app_name> (your compiled application)

## Protocol

1. Write your application *app_name.cpp* and compile on the cluster.
2. Take *openmpiscript* from its directory
    
    ```bash
    # copy openmpiscript from the root directory
    
    cp /OpenmpiScript/openmpiscript /home/<your_id>/openmpiscript
    ```
    
3. Create in your folder <your_app_name>.job
    
    ```bash
    universe = parallel
    
    executable = openmpiscript
    
    arguments = <you_app_name>
    
    should_transfer_file = yes
    
    transfer_input_file = <your_app_name>
    
    when_to_transfer_output = on_exit_or_evict
    
    output = logs/out.$(NODE)
    error = logs/err.$(NODE)
    log = logs/log
    
    machine_count = <number_of_nodes_you_need>
    request_cpus = <cores_needed_per_node>
    ```
    
    1. Launch the job