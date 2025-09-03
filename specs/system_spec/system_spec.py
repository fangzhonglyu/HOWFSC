import yaml
import math

class SystemSpec:
    def load_from_yaml(self, spec_yml: str):
        with open(spec_yml, 'r') as file:
            self.spec = yaml.safe_load(file)
        self.name = self.spec['name']
        self.n_channels = self.spec['n_channels']
        self.iwa = self.spec['iwa']
        self.owa = self.spec['owa']
        self.actuator_per_side = self.spec['actuator_per_side']
        self.pixels_per_side = self.spec['pixels_per_side']
        self.l_d_per_pix = self.spec['l_d_per_pix']
        self.bit_depth = self.spec['bit_depth']
        self.n_probes = self.spec.get('n_probes', 7)

    def compute_params(self):
        assert self.spec is not None, "Specification not loaded. Call load_spec() first."
        self.dark_hole_area = round(math.pi * (self.owa**2 - self.iwa**2))
        self.n_pixels = round(self.dark_hole_area / (self.l_d_per_pix ** 2))
        self.n_actuators = round(2 * math.pi * (self.actuator_per_side / 2)**2)
        self.dof = 2 * self.n_channels * self.n_pixels
        #self.jacobian_size_GB = self.n_actuators * self.dof * self.bit_depth / 8 / 1e9

    def __init__(self, spec_yml: str):
        self.load_from_yaml(spec_yml)
        self.compute_params()

