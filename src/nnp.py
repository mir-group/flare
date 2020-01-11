import numpy as np
import ace
import torch

class NNP(torch.nn.Module):
    def __init__(self, nos, layers, input_size, activation,
                 descriptor_calculator, descriptor_method, cutoff,
                 optimizer=torch.optim.SGD, optimizer_kwargs={"lr" : 0.01},
                 criterion=torch.nn.MSELoss()):
        super(NNP, self).__init__()

        # Set descriptor values.
        self.descriptor_calculator = descriptor_calculator
        self.descriptor_method = descriptor_method
        self.cutoff = cutoff

        # Create species nets.
        for n in range(nos):
            setattr(self, "spec" + str(n),
                    SpeciesNet(layers, input_size, activation))

        # Set optimizer and loss function.
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        self.criterion = criterion

    def get_torch_descriptor(self, local_environment):
        # Calculate descriptor.
        getattr(self.descriptor_calculator,
                self.descriptor_method)(local_environment)
        descriptor = \
            torch.tensor(self.descriptor_calculator.descriptor_vals).double()
        descriptor.requires_grad = True
        coordinate_gradient = \
            torch.from_numpy(self.descriptor_calculator.descriptor_force_dervs)
        strain_gradient = \
            torch.from_numpy(self.descriptor_calculator
                             .descriptor_stress_dervs)

        return descriptor, coordinate_gradient, strain_gradient

    def predict_local_E(self, local_environment):
        descriptor, _, _ = self.get_torch_descriptor(local_environment)

        # Forward pass.
        spec = local_environment.central_species
        local_energy = getattr(self, "spec" + str(spec)).forward(descriptor)

        return local_energy
    
    def predict_local_F(self, local_environment):
        f_tens = torch.zeros(3 * local_environment.noa).double()
        descriptor, desc_grad_torch, _ = \
            self.get_torch_descriptor(local_environment)
        spec = local_environment.central_species
        local_energy = getattr(self, "spec" + str(spec)).forward(descriptor)
        net_grad = \
            torch.autograd.grad(local_energy, descriptor, create_graph=True)[0]
        f_tens = torch.mv(desc_grad_torch, net_grad)

        return f_tens

    def predict_local_EF(self, local_environment):
        # Initialize energy/force tensor.
        ef_tens = torch.zeros(1 + 3 * local_environment.noa).double()

        # Calculate descriptor.
        descriptor, desc_grad_torch, _ = \
            self.get_torch_descriptor(local_environment)

        # Forward pass to get local energy.
        spec = local_environment.central_species
        local_energy = getattr(self, "spec" + str(spec)).forward(descriptor)

        # Compute partial forces.
        net_grad = \
            torch.autograd.grad(local_energy, descriptor, create_graph=True)[0]

        # Store energy and partial forces.
        ef_tens[0] = local_energy
        ef_tens[1:] = torch.mv(desc_grad_torch, net_grad)

        return ef_tens
    
    def predict_local_EFS(self, local_environment):
        # Initialize energy/force/stress tensor.
        no_force_comps = 3 * local_environment.noa
        efs_tens = torch.zeros(1 + no_force_comps + 6).double()

        # Calculate descriptor.
        descriptor, coordinate_gradient, strain_gradient = \
            self.get_torch_descriptor(local_environment)
        
        # Forward pass to get local energy.
        spec = local_environment.central_species
        local_energy = getattr(self, "spec" + str(spec)).forward(descriptor)

        # Compute partial forces.
        net_grad = \
            torch.autograd.grad(local_energy, descriptor, create_graph=True)[0]
        frcs = torch.mv(coordinate_gradient, net_grad)

        # Compute partial stress.
        struc_vol = local_environment.structure_volume
        stress = -torch.mv(strain_gradient, net_grad) / struc_vol

        # Store energy, partial forces, and partial stress.
        efs_tens[0] = local_energy
        efs_tens[1:1+no_force_comps] = frcs
        efs_tens[-6:] = stress

        return efs_tens

    def predict_E(self, structure):
        energy = torch.zeros(1).double()

        for count in range(len(structure.species)):
            environment = ace.LocalEnvironment(structure, count, self.cutoff)
            energy += self.predict_local_E(environment)

        return energy

    def predict_F(self, structure):
        noa = len(structure.species)
        f_tens = torch.zeros(3 * noa).double()

        for count in range(noa):
            environment = ace.LocalEnvironment(structure, count, self.cutoff)
            f_tens += self.predict_local_F(environment)
        
        return f_tens

    def predict_EF(self, structure):
        # Initialize energy/force tensor.
        noa = len(structure.species)
        ef_tens = torch.zeros(1 + 3 * noa).double()

        for count in range(noa):
            environment = ace.LocalEnvironment(structure, count, self.cutoff)
            ef_tens += self.predict_local_EF(environment)

        return ef_tens

    def predict_EFS(self, structure):
        noa = len(structure.species)
        efs_tens = torch.zeros(1 + 3 * noa + 6).double()

        for count in range(noa):
            environment = ace.LocalEnvironment(structure, count, self.cutoff)
            efs_tens += self.predict_local_EFS(environment)
        
        return efs_tens

    def update_weights(self, structure, target):
        # Zero the gradients.
        self.optimizer.zero_grad()

        # Check target type.
        noa = len(structure.species)
        if len(target) == 1:
            pred = self.predict_E(structure)
        elif len(target) == 3 * noa:
            pred = self.predict_F(structure)
        elif len(target) == 1 + 3 * noa:
            pred = self.predict_EF(structure)
        elif len(target) == 1 + 3 * noa + 6:
            pred = self.predict_EFS(structure)
        else:
            raise Exception('Target tensor has the wrong length.')

        # Compute loss and udpate weights.
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

class SpeciesNet(torch.nn.Module):
    def __init__(self, layers, input_size, activation):
        super(SpeciesNet, self).__init__()

        # Store linear layers of the network.
        self.layer_count = len(layers) + 1

        setattr(self, "lin0", torch.nn.Linear(input_size, layers[0]).double())

        for n in range(1, len(layers)):
            # Connect previous hidden layer to current hidden layer.
            setattr(self, "lin"+str(n),
                    torch.nn.Linear(layers[n-1], layers[n]).double())

        # Connect final hidden layer to the output.
        setattr(self, "lin"+str(len(layers)),
                torch.nn.Linear(layers[-1], 1).double())

        # Set the activation function.
        self.activation = activation

    def forward(self, descriptor):
        x = descriptor
        for n in range(self.layer_count - 1):
            lin = getattr(self, "lin" + str(n))
            x = self.activation(lin(x))
        return getattr(self, "lin" + str(self.layer_count - 1))(x)
