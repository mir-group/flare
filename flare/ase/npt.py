from ase.md.npt import NPT


class NPT_mod(NPT):
    def stresscalculator(self):
        stress = self.atoms.get_stress(include_ideal_gas=True)

        # Clear the calculator before forces are computed on new positions.
        self.atoms.calc.results = {}

        return stress
