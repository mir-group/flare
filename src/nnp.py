import numpy as np
import ace
import torch

class NNP(torch.nn.Module):
    def __init__(self, nos, layers, input_size, activation,
                 descriptor_calculator, descriptor_method, cutoff):
        super(NNP, self).__init__()

        self.descriptor_calculator = descriptor_calculator
        self.descriptor_method = descriptor_method
        self.cutoff = cutoff

        for n in range(nos):
            setattr(self, "spec" + str(n),
                    SpeciesNet(layers, input_size, activation))

    def get_torch_descriptor(self, local_environment):
        # Calculate descriptor.
        getattr(self.descriptor_calculator,
                self.descriptor_method)(local_environment)
        descriptor = \
            torch.tensor(self.descriptor_calculator.descriptor_vals).double()
        descriptor.requires_grad = True
        descriptor_grad = \
            torch.from_numpy(self.descriptor_calculator.descriptor_force_dervs)

        return descriptor, descriptor_grad

    def predict_local_E(self, local_environment):
        descriptor, _ = self.get_torch_descriptor(local_environment)

        # Forward pass.
        spec = local_environment.central_species
        local_energy = getattr(self, "spec" + str(spec)).forward(descriptor)

        return local_energy

    def predict_local_EF(self, local_environment):
        # Initialize energy/force tensor.
        ef_tens = torch.zeros(1 + 3 * local_environment.noa)

        # Calculate descriptor.
        descriptor, desc_grad_torch = \
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
        efs_tens = torch.zeros(1 + 3 * local_environment.noa + 6)

        # Calculate descriptor.
        descriptor, desc_grad_torch = \
            self.get_torch_descriptor(local_environment)
        
        # Forward pass to get local energy.
        spec = local_environment.central_species
        local_energy = getattr(self, "spec" + str(spec)).forward(descriptor)

        # Compute partial forces.
        net_grad = \
            torch.autograd.grad(local_energy, descriptor, create_graph=True)[0]
        
        # Store energy, partial forces, and partial stress.

        pass

    def predict_E(self, structure):
        energy = torch.zeros(1).double()

        for count in range(len(structure.species)):
            environment = ace.LocalEnvironment(structure, count, self.cutoff)
            energy += self.predict_local_E(environment)

        return energy

    def predict_EF(self, structure):
        pass

    def predict_EFS(self, structure):
        pass

    def update(self, structure, labels):
        pass

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
