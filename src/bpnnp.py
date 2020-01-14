import numpy as np
import ace
import torch
import copy

class BPNNP(torch.nn.Module):
    def __init__(self, nos, layers, input_size, activation,
                 descriptor_calculator, descriptor_method, cutoff,
                 optimizer=torch.optim.SGD, optimizer_kwargs={"lr" : 0.01},
                 criterion=torch.nn.MSELoss()):
        super(BPNNP, self).__init__()

        # Set descriptor values.
        self.descriptor_calculator = descriptor_calculator
        self.descriptor_method = descriptor_method
        self.cutoff = cutoff

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

    def get_local_energy(self, descriptor):
        x = descriptor
        for n in range(self.layer_count - 1):
            lin = getattr(self, "lin" + str(n))
            x = self.activation(lin(x))
        return getattr(self, "lin" + str(self.layer_count - 1))(x)

    def predict_local_F(self, local_environment):
        f_tens = torch.zeros(3 * local_environment.noa).double()
        descriptor, desc_grad_torch, _ = \
            self.get_torch_descriptor(local_environment)
        local_energy = self.get_local_energy(descriptor)
        net_grad = \
            torch.autograd.grad(local_energy, descriptor, create_graph=True)[0]
        f_tens = -torch.mv(desc_grad_torch, net_grad)

        return f_tens

    def predict_F(self, structure):
        noa = len(structure.species)
        f_tens = torch.zeros(3 * noa).double()

        for count in range(noa):
            environment = ace.LocalEnvironment(structure, count, self.cutoff)
            f_tens += self.predict_local_F(environment)

        return f_tens

    def update_weights(self, structure, target):
        # Zero the gradients.
        self.optimizer.zero_grad()

        # Check target type.
        noa = len(structure.species)
        if len(target) == 3 * noa:
            pred = self.predict_F(structure)
        else:
            raise Exception('Target tensor has the wrong length.')

        # Compute loss and udpate weights.
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()