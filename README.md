# Fermionic Neural Network: PHYS 449 Final Project
This projects goal is to replicate the algorithm and experiments defined in 
"Ab-Initio Solution of the Many-Electron Schroedinger Equation with Deep Neural Networks", David Pfau, James S. Spencer, Alex G de G Matthews and W.M.C. Foulkes, Phys. Rev. Research 2, 033429 (2020).
\
\
The objective of the FermiNet is to create the optimal trial Ansatz for spin systems that obeys 
Fermi-Dirac statistics. The FermiNet uses no data other than atomic positions and charges.


## Running The FermiNet
To run and train the FermiNet, use the `driver.py` file, run
```
python driver.py --name NAME --length LENGTH -- param PARAM.json -res-path RESULT PATH -n N -v V
```
When `python driver.py --h` is used, returns
```sh
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
When `python driver.py` is ran a folder with the date and time is created to store the following,
* _state
* _state_dict
* command_line_arguments.txt
* loss_storage.txt
* time.txt

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
