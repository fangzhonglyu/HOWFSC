import yaml

class ComputeSpec():

    def parse_flops(flops_str: str):
        if 'PFLOPs' in flops_str:
            return float(flops_str.replace('PFLOPs', '')) * 1e15
        elif 'TFLOPs' in flops_str:
            return float(flops_str.replace('TFLOPs', '')) * 1e12
        elif 'GFLOPs' in flops_str:
            return float(flops_str.replace('GFLOPs', '')) * 1e9
        elif 'MFLOPs' in flops_str:
            return float(flops_str.replace('M', '')) * 1e6
        
    def parse_mem_bw(mem_bw_str: str):
        if 'TB/s' in mem_bw_str:
            return float(mem_bw_str.replace('TB/s', '')) * 1e12
        elif 'GB/s' in mem_bw_str:
            return float(mem_bw_str.replace('GB/s', '')) * 1e9
        elif 'MB/s' in mem_bw_str:
            return float(mem_bw_str.replace('MB/s', '')) * 1e6
        elif 'KB/s' in mem_bw_str:
            return float(mem_bw_str.replace('KB/s', '')) * 1e3

    def __init__(self, spec_yml: str):
        self.spec_yml = spec_yml
        self.load_spec()

    def load_spec(self):
        with open(self.spec_yml, 'r') as file:
            self.spec = yaml.safe_load(file)
            self.name = self.spec['name']
            self.fp32_FLOPs = ComputeSpec.parse_flops(self.spec['fp32_FLOPs'])
            self.fp64_FLOPs = ComputeSpec.parse_flops(self.spec.get('fp64_FLOPs', '0 GFLOPs'))
            self.mem_bw = ComputeSpec.parse_mem_bw(self.spec['mem_bw'])
