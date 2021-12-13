# Fermionic Neural Network: PHYS 449 Final Project
This projects goal is to replicate the algorithm and experiments defined in 
"Ab-Initio Solution of the Many-Electron Schroedinger Equation with Deep Neural Networks", David Pfau, James S. Spencer, Alex G de G Matthews and W.M.C. Foulkes, Phys. Rev. Research 2, 033429 (2020).
\
\
The objective of the FermiNet is to create the optimal trial Ansatz for spin systems that obeys 
Fermi-Dirac statistics. The FermiNet uses no data other than atomic positions and charges.


## Running The FermiNet
To run and train the FermiNet, use the `driver.py` file.
The simplest run is as follows:
```
python driver.py
```
which will attempt to train the network on the hydrogen atom using all the default parameters.


You can adjust some of the hyperparameters and additional arguments like so
```
python driver.py --name NAME --length LENGTH -res-path RESULT_PATH -n N -v V
```

All command line arguments can be seen as follows `python driver.py --h`:
```
-h, --help           show this help message and exits
--name system name   the name of the QM system to evalute (default: hydrogen)
--length length of chain
                     if using a hydrogen chain, how long the chain is (defult:1)
--param param.json   file name for training parameters (default: param.json)
-res-path results_dir 
                     path to save the plots at (default: results)
-n number_of_replicas
                     number of replica state vectors for mc to propagate (default: 7)
-v N                 verbosity (set to 2 for full debugging) (default: 1)
-device device_name  specify a specific device for pyTorch to use (default: cuda)
-epochs num_epochs   number of epochs to run for (default: 200)
-lr learning_rate    learning rate for the optimiser (default: 0.001)
-plot plot_control   will run simple_plotting() if greater than 0 (default: 0)
```

### Parameters
The goal was to allow for specification of all hyperparameters through `param.json` although it was not implemented.
Three hyperparameters: `learning_rate`, `num_epochs`, `number_of_replicas`, can be changed from the command line. (note that `batch_size` and `number_of_replicas` are synonyms).
All other hyperparameters are specified in `flags.py` and must be changed manually.

+ Network hyperparameters (see [lines 148-150 in `flags.py`](https://github.com/ngraymon/DNN-SE/blob/main/flags.py#L148))
  + `flags.determinants`    (default `16`)
  determinants used by `fnn.py`
  + `flags.hidden_units`    (default `[[32, 4],  [32, 4],  [32, 4],  [32, 4]]`)
  the size of the single and double stream layers

+ Monte Carlo hyperparameters  (see [lines 113-119 in `flags.py`](https://github.com/ngraymon/DNN-SE/blob/main/flags.py#L113))
Where the distribution parameters specify the shape of the normal distribution that new states are drawn from
  + `flags.mcmc_offset`             (default `0.0`)  μ/mean/offsets
  + `flags.mcmc_move_width`         (default `0.02`) σ/width/standard-deviation
  + `flags.mcmc_init_means`         (default `0.8`)  **initial** μ/mean/offsets  (is broadcast to a `(number_of_electrons, 3)` shaped tensor)
  + `flags.mcmc_initial_stddev`     (default `''`)  **initial** σ/width/standard-deviation (is ignored, see below)
  + `flags.batch_size`              (default `100`)  how many walkers/replicas/independent Markov-Chains to use during MC propagation

### Note
In the current implementation the parameter `flags.mcmc_init_means` is not used. 
The **initial** μ/mean/offsets are not a hyperparameter but are instead derived from the initial (x,y,z) positions of the electrons in the system.
By default they are instantiated at [line 627 in `driver.py`](https://github.com/ngraymon/DNN-SE/blob/main/driver.py#L627) using the `generate_electron_position_vector()` function. You could manually set these at either [line 622 in `driver.py`](https://github.com/ngraymon/DNN-SE/blob/main/driver.py#L622) or [line 540 in `driver.py`](https://github.com/ngraymon/DNN-SE/blob/main/driver.py#L540), although I wouldn't recommend this.

## Supported systems
The provided system name is used to determine what system to train inside the function [`prepare_molecule_and_spins` at line 814](https://github.com/ngraymon/DNN-SE/blob/main/driver.py#L814). Both the user's input name and the system names are converted to lower case before comparing them.

The following atoms are supported:
+ Hydrogen:     `hydrogen`    
+ Helium:       `helium`      
+ Lithium:      `lithium`     
+ Beryllium:    `beryllium`   
+ Boron:        `boron`       
+ Carbon:       `carbon`      
+ Nitrogen:     `nitrogen`    
+ Oxygen:       `oxygen`      
+ Flourine:     `flourine`    
+ Neon:         `neon`        

The following diatomics are supported:
+ Lithium Hydride:   `LiH`
+ Lithium gas:       `Li2`
+ Carbon monoxide:   `CO`
+ Nitrogen gas:      `N2`

The following molecules are supported:
+ Hydrogen Chain:       `chain`     (the length is specified by the `--length` input)
+ Methane:              `methane`
+ Ammonia:              `ammonia`
+ Ethylene/Ethene:      `ethylene`
+ Bicyclobutane:        `bicyclobutane`
+ 4-Hydrogen-Circle:    `h4circle`

All parameters of the systems are laid out in `systems.py` using the functions:
+ `hydrogen_chains` [line 141](https://github.com/ngraymon/DNN-SE/blob/main/system.py#L141)
+ `diatomic`        [line 162](https://github.com/ngraymon/DNN-SE/blob/main/system.py#L162)
+ `helium`          [line 190](https://github.com/ngraymon/DNN-SE/blob/main/system.py#L190)
+ `methane`         [line 199](https://github.com/ngraymon/DNN-SE/blob/main/system.py#L199)
+ `ammonia`         [line 212](https://github.com/ngraymon/DNN-SE/blob/main/system.py#L212)
+ `ethylene`        [line 224](https://github.com/ngraymon/DNN-SE/blob/main/system.py#L224)
+ `bicyclobutane`   [line 238](https://github.com/ngraymon/DNN-SE/blob/main/system.py#L238)
+ `h4_circle`       [line 256](https://github.com/ngraymon/DNN-SE/blob/main/system.py#L256)


## File Output

For every execution of `python driver.py` a folder of the format `results_<month>_<day>_##h_##m_##s` is created at the path provided using `-res-path`.

Inside that folder the following files are created during training/execution:
* `command_line_arguments.txt`  (before training occurs we save the provided arguments to a text file)
* `time.txt`                    (simply record the time it takes to train using `time.time()`)
* `loss_storage.txt`            (the per-epoch loss values stored in order to a simple text file)
* `_state`                      (the model saved after training using `torch.save` see [line 219 in `fnn.py`](https://github.com/ngraymon/DNN-SE/blob/main/fnn.py#L219) )
* `_state_dict`                 (the `state_dict` saved after training using `torch.save` see [line 228 in `fnn.py`](https://github.com/ngraymon/DNN-SE/blob/main/fnn.py#L228))

If the `plot_control` was greater than 0 the code will also attempt to generate one or more plots using the function [`simple_plotting()` at line 774 in `driver.py`](https://github.com/ngraymon/DNN-SE/blob/main/driver.py#L774).
For all systems it will plot the progress of the loss value during training. If the system is helium or the 4-Hydrogen-circle it will also create an additional plot trying to emulate Fig 4 and 11 from the paper.
The plots will be save to the path `results_dir/flags.plot_path/`. You can change the plotting path at [line 165 in `flags.py`](https://github.com/ngraymon/DNN-SE/blob/main/flags.py#L165) 


## Runtime Output / Verbosity
The user can specify three types of runtime output using the `-v` command line argument: [_Minimal_ (default), _Runtime_, _Debug_](https://github.com/ngraymon/DNN-SE/blob/main/driver.py#L906).
As an example here is typical output for the default system Hydrogen:
### _Minimal_ (`-v 1`)

Prints the loss every ~ 5 seconds of training
```
The system input is as follows:
  Atom 0: Atom(symbol='H', number=1, coords=(0.0, 0.0, 0.0), charge=1)
  spins = (0, 1)

12:45:40 [ INFO]       driver:                           main:( 685): Running on device: cpu
12:45:45 [ INFO]        train:                          train:(  62): Epoch: (38         / 200   )
12:45:50 [ INFO]        train:                          train:(  62): Epoch: (75         / 200   )
12:45:56 [ INFO]        train:                          train:(  62): Epoch: (112        / 200   )
12:46:01 [ INFO]        train:                          train:(  62): Epoch: (149        / 200   )
12:46:06 [ INFO]        train:                          train:(  62): Epoch: (186        / 200   )
12:46:08 [ INFO]       driver:                           main:( 758): Training completed         [27.3447 s]
12:46:08 [ INFO]          fnn:                           save:( 221): FermiNet model saved to file:
~/DNN-SE/results/results_Dec_13_12h_58m_00s/_state
To load, use 'model = torch.load(<path>)'
12:46:08 [ INFO]          fnn:                           save:( 230): FermiNet model `state_dict` saved to file:
~/DNN-SE/results/results_Dec_13_12h_58m_00s/_state_dict
To load, use 'model = model.load_state_dict(torch.load(<path>))'

Success!    Final loss: -2.488679 after 200 epochs
```
### _Runtime_ (`-v 2`)

The same as _Minimal_ but also prints the loss every single epoch.
Additional lines look like this:
```
01:08:33 [RUNTIME]        train:                          train:(  86): Loss for epoch 0: loss = tensor(-0.0349, dtype=torch.float64, grad_fn=<MeanBackward0>)
01:08:33 [RUNTIME]        train:                          train:(  86): Loss for epoch 1: loss = tensor(-0.0836, dtype=torch.float64, grad_fn=<MeanBackward0>)
...
...
01:09:01 [RUNTIME]        train:                          train:(  86): Loss for epoch 199: loss = tensor(-2.4887, dtype=torch.float64, grad_fn=<MeanBackward0>)
```
### _Debug_ (`-v 3`)

The same as above but also prints all `log.debug()` statements.
This should only be used when debugging and it is best to only run for two epochs for ease of scrolling through the output.
It can be used to check the dimensionality of the stream/layer objects in the network.
```
01:18:02 [DEBUG]          fnn:                       __init__:( 424): Initializing a FermiLayer

01:18:02 [DEBUG]          fnn:                       __init__:( 428): hidden_layer_dimensions = ([4, 4], [32, 4])
01:18:02 [DEBUG]          fnn:                       __init__:( 543): self.v_matrices.shape = torch.Size([1, 1, 12, 32])
01:18:02 [DEBUG]          fnn:                       __init__:( 544): self.b_vectors.shape = torch.Size([1, 1, 32])
01:18:02 [DEBUG]          fnn:                       __init__:( 545): self.w_matrices.shape = torch.Size([1, 1, 1, 4, 4])
01:18:02 [DEBUG]          fnn:                       __init__:( 546): self.c_vectors.shape = torch.Size([1, 1, 1, 4])
01:18:02 [DEBUG]          fnn:                       __init__:( 424): Initializing a FermiLayer

01:18:02 [DEBUG]          fnn:                       __init__:( 428): hidden_layer_dimensions = ([32, 4], [32, 4])
01:18:02 [DEBUG]          fnn:                       __init__:( 543): self.v_matrices.shape = torch.Size([1, 1, 68, 32])
01:18:02 [DEBUG]          fnn:                       __init__:( 544): self.b_vectors.shape = torch.Size([1, 1, 32])
01:18:02 [DEBUG]          fnn:                       __init__:( 545): self.w_matrices.shape = torch.Size([1, 1, 1, 4, 4])
01:18:02 [DEBUG]          fnn:                       __init__:( 546): self.c_vectors.shape = torch.Size([1, 1, 1, 4])
01:18:02 [DEBUG]          fnn:                       __init__:( 424): Initializing a FermiLayer
```

It also shows more information about steps such as the initial state of the system before training.
Using Methane as an example: (where `[1, 0]` means 1 up and 0 down electrons were assigned to that particular atom)
```
Electrons assigned:
    C: [3, 3], H: [1, 0], H: [0, 1], H: [1, 0], H: [0, 1].
01:18:02 [DEBUG]       driver: generate_electron_position_vector:( 380):
up_spin_list:
    array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    array([1.18886, 1.18886, 1.18886])
    array([], dtype=float64)
    array([ 1.18886, -1.18886, -1.18886])
    array([], dtype=float64)
01:18:02 [DEBUG]       driver: generate_electron_position_vector:( 381):
down_spin_list:
    array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    array([], dtype=float64)
    array([-1.18886, -1.18886,  1.18886])
    array([], dtype=float64)
    array([-1.18886,  1.18886, -1.18886])
01:18:02 [DEBUG]       driver: generate_electron_position_vector:( 390):
walker tensor:
    electron 1 : [0. 0. 0.]
    electron 2 : [0. 0. 0.]
    electron 3 : [0. 0. 0.]
    electron 4 : [1.18886 1.18886 1.18886]
    electron 5 : [ 1.18886 -1.18886 -1.18886]
    electron 6 : [0. 0. 0.]
    electron 7 : [0. 0. 0.]
    electron 8 : [0. 0. 0.]
    electron 9 : [-1.18886 -1.18886  1.18886]
    electron 10 : [-1.18886  1.18886 -1.18886]
```


## Contributors
* Andrew Francey
* Jean-Baptiste Valentin
* Matthew Waters
* Neil Goerge Raymond
* Willian Archer

## Original GitHub repository
The FermiNet was created by David Pfau, James S. Spencer, Alex G de G Matthews 
and W.M.C. Foulkes. Described in 
This github is based off the original github which can be found at,
https://github.com/deepmind/ferminet
