'''
Hamiltonian module

This module contains the functions for constructing Hamiltonians and setting 
up the VMC calculations.

Import this module as:
    import Hamiltonian as H
'''

# Import
import numpy as np
import torch


def kinetic_from_log(f,x):
    '''
    Computes the kinetic energy from the log of |psi|, 
    the -1/2 \nabla^2 \psi / \psi.

    Parameters
    ----------
    f : Torch Tensor
        The log psi function
    x : Torch Tensor
        Tensor for the coordinates.

    Returns
    -------
    The kinetic energy function.
    '''
    
    df = torch.autograd(f,x)[0]
    lapl_tensor = []
    for i in range(x.shape[1]):
        lapl_elem = torch.autograd(torch.unsqueeze(df[...,i], -1),x)
        lapl_tensor.append(lapl_elem)
    lapl_tensor = torch.tensor(lapl_tensor)
    lapl = torch.sum(lapl_tensor) + torch.sum(df**2)
    return -0.5*torch.unsqueeze(lapl,-1)
    
        
    
    
def operators(atoms, nelectrons, potential_epsilon=0.0):
    '''
    Creates the kinetic and potential operators of the Hamiltonian in atomic 
    units.

    Parameters
    ----------
    atoms : List of objects
        A list of objects from the Atom class.
    nelectrons : Integer
        The number of electrons in the system.
    potential_epsilon : Argument
        Epsilon used to smooth the divergence of the 1/r potential near the
        origin. The default is 0.0.

    Returns
    -------
    Functions for the kinetic and potential energy as a pytorch operator
    '''
    
            
    def smooth_norm(x):
        '''
        Function used to smooth out an instabilities when x approaches 0 in 
        functions involving 1/x.
        
        Parameters
        ----------
        x : Torch tensor of float points
          Values that approaches 0
        
        Returns
        -------
        The norm of the tensors rows.
        '''
        
        # If their is no instability then return the norm of x
        if potential_epsilon == 0: 
            return torch.norm(x,dim=1,keepdim=True)
        # Else we add the epsilon term then return the norm.
        else: 
            return torch.sqrt(torch.sum(x**2 + potential_epsilon**2, 
                                        dim=1,keepdim=True))
    
    
    def nuclear_potential(e_positions):
        '''
        Calculates the nuclear potential for set of electron positions.

        Parameters
        ----------
        e_positions : Torch tensor
            A tensor of electron positions.

        Returns
        -------
        The potential between the nuclues and the electrons.
        '''
        
        # the potental for each nucleus
        v = []
        # Add up all the potentials between all the nucleus and their electorns
        for atom in atoms:
            charge = torch.tensor(atom.charge, dtype = e_positions[0].dtype)
            coords = torch.tensor(atom.coord, dtype = e_positions[0].dtype)
            v.extend([-charge / smooth_norm(coords - x) for x in e_positions])
        v = torch.tensor(v)
        return torch.sum(v)
    
    
    def electronic_potential(e_positions):
        '''
        Calculates the electric potential for the set of electron positions.

        Parameters
        ----------
        e_positions : Torch tensor
            A tensor of electron positions.

        Returns
        -------
        The potential between the electrons.
        '''
        
        # If there is more the one electron in the system.
        if len(e_positions) > 1:
            v = []
            for (i,ri) in enumerate(e_positions):
                v.extend([1/ smooth_norm(ri - rj) for rj in xs[i + 1:]])
            v = torch.tensor(v)
            return torch.sum(v)
        else:
            return torch.tensor(0.0)
  
    
    def nuclear_nuclear(dtype):
        '''
        Calculates the potential between all the nucleus' in the system.

        Parameters
        ----------
        dtype : Torch Type
            The type of the tensor to be returned.

        Returns
        -------
        Torch Tensor for the potential of the nucleus'.
        '''
        
        # The nucleus to nucleus potential
        vnn = 0.0   
    
        # Loops over all the combinations of atoms in the system
        for i, atom_i in enumerate(atoms):
            for atom_j in atomes[i+1:]:
                # Charge of atom i an atom j.
                qij = float(atom_i.charge * atom_j.charge)
                # Add the potential between atom i and atom j.
                vnn += qij / np.linalg.norm(atom_i.coords_array 
                                            - atom_j.coords_array)

        return torch.tensor([vnn],dtype = dtype)
        
    
    def potential(positions):
        '''
        Splits the tensor x into the tensor xs for the electron positions. 
        Then compute the potntials and adds them together to return the total
        potential.

        Parameters
        ----------
        positions : Torch Tensor
            The position tensor for the electrons and nucleus'.

        Returns
        -------
        The total potential
        '''
        
        e_positions = torch.split(positions,nelectrons,dim=1)
        
        return (nuclear_potential(e_positions) 
              + electronic_potential(e_positions)
              + nuclear_nuclear(e_positions.dtype))
        



def exact_hamiltonian(atoms, nelectrons, potential_epsilon = 0.0):
    '''
    Evaluates the exact hamiltonian of a system.

    Parameters
    ----------
    atoms : Object
        The object that contains the atoms properties.
    nelectrons : Integer
        The number of electrons in the system.
    potential_epsilon : Float, optional
        Value to fix instability around 1/r. The default is 0.0.

    Returns
    -------
    The functions that generates the wavefunction and the hamiltonian op.
    '''
    
    # The kinetic and the potential functions.
    k_fn, v_fn = operators(atoms, nelectronsm potential_epsilon = 0.0)
    
    def _hamiltonian(f, x):
        logpsi, signpsi = f(x)
        psi = torch.exp(logpsi) * signpsi
        hpsi = psi * (k_fn(logpsi, positons) + v_fn(positions))
        return psi, hpsi
    
    return _hamiltonian




def r12_features(e_post, atoms, nelectrons, keep_pos=True, flatten=False,
                 atomic_coords=False):
    '''
    Adds physically-motivated features depending upon electron distances.
    
    The tensor of electron positions is extended to include the distance of
    each electron to each nucleus and distance between each pair of electrons.

    Parameters
    ----------
    x : Tensor | shape(batch_size,nelectrons*ndim) 
                 or (batch_size, nelectrons, ndim)
        Electron positions, ndim is the dimensionality of the system.
    atoms : List of object
        list of atom objects for each atom in the system.
    nelectrons : Integer
        The number of electrons.
    keep_pos : Boolian, optional
        If True includes the original electron positions in the output.
        The default is True.
    flatten : Boolian, optional
        If True, returns the distances as a flat vector for each element of 
        the batch If False, return the atom-electron distnaces and electron-
        electron distances each as 3D arrays. 
        The default is False.
    atomic_coords : Boolian, optional
        If True, replace the original positon of the electrons with the
        position of the electrons relative to all atoms.
        The default is False.

    Returns
    -------
    If flatten is true, keep_pos is true and atomic_coords is false:
        Tensor of shape (batch_size, ndim*Ne + Ne*Na + Ne(Ne-1)/2), where Ne
        (Na) is the number of electrons (atoms). The first ndim*Ne terms are
        the original x, the next Ne terms are |x_i - R_1|, where R_1 is the 
        position of the first nucleus (and so on for each atom), and the
        remaining terms are |x_i - x_j| for each (i,j) pair, where i and j run
        over all electrons with i varied slowest.
    If flatten is true and keep_pos is false: 
        It does not include the first ndim*Ne features.
    If flatten is false and keep_pos is false:
        Tensors of shape (batch_size, Ne, Na) and (batch_size, Ne, Ne)
    If flatten is false and keep_pos is true:
        Same as above, and also a tensor of size (batch_size, Ne, ndim)
    If atomic_coords is true:
        The same as if keep_pos is true, except the ndim*Ne coordinates
        corresponding to the original positions are replaced by ndim*Ne*Na
        coordinates corresponding to the different from each electron position
        to each atomic position.
    '''
    
    ## Converts e_post to the same size
    if len(x.shape) == 2:
        e_posts = torch.reshape(e_post,[e_post.shape[0],nelectrons,-1])
    else:
        e_posts = e_post
    
    ## Coordinates of the nucleus
    coords = torch.stack([torch.tensor([atom.coords,dtype=x.dtype.base_dtype])]
                         for atom in atoms)
    
    coords = torch.unsqueeze(torch.unsqueeze(coords, 0), 0)
    
    ## Converts the absolute electron positions to positions relative to the
    ## nucleus'
    e_posts_atomic = torch.unsqueeze(e_posts, 2) - coords
    
    ## The distance between the electron and the nucleus'
    r_ae = torch.norm(e_posts_atomic, dim=-1)
    
    ## The distance between the electron pairs.
    r_ee = np.zeros((nelectrons, nelectrons), dtype=object)
    for i in range(nelectrons):
        for j in range(i+1, nelectrons):
            r_ee[i,j] = torch.norm(e_posts[:,i,:]-e_posts[:,j,:], 
                                   dim=1, keepdim=True)
            
    if flatten:
        r_ae = torch.reshape(r_ae, [r_ae.shape[0],-1])
        if nelectrons > 1:
            r_ee = torch.cat(r_ee[np.triu_indices(nelectrons,k=1)].tolist(),
                             axis=1)
        else:
            r_ee = torch.zeros([r_ae.shape[0],0])
        
        if keep_pos:
            if atomic_coords:
                e_posts_atomic = torch.reshape(e_posts_atomic,
                                               [e_posts_atomic[0],-1])
                return torch.cat([r_ae, r_ee, e_posts_atomic], dim=1)
            else:
                return torch.cat([r_ae, r_ee, e_post], dim=1)
        else:
            return torch.cat([r_ae,r_ee],dim=1)
    else:
        zeros_like = torch.zeros((e_posts.shape[0],1), 
                                 dtype=x.dtype.base_dtype)
        
        for i in range(nelectrons):
            r_ee[i,i] = zeros_like
            for j in range(i):
                r_ee[i,j] = r_ee[j,i]
        r_ee = torch.transpose(torch.stack(r_ee.tolist(),[2,0,1,3]))
        if keep_pos:
            if atomic_coords:
                return r_ae, r_ee, e_posts_atomic
            else:
                return r_ae, r_ee, e_post
        else:
            r_ae, r_ee
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    