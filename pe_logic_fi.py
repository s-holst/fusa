from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from kyupy import log, cdiv, verilog, Timers, logic_sim
from kyupy.logic import unpackbits, packbits
from kyupy.techlib import SAED90

class FMAlogic(logic_sim.LogicSim):
    def __init__(self, circuit, sims=1024):
        super().__init__(circuit, sims, m=2, c_reuse=False, strip_forks=False)

        self.circuit = circuit
        self.timers = Timers()

        self.a_locs = self.circuit.s_locs('activation_reg')
        self.b_locs = self.circuit.s_locs('weight_reg')
        self.s_locs = self.circuit.s_locs('i_sum')
        self.r_locs = self.circuit.s_locs('o_sum_reg')

        assert len(self.a_locs) == 8
        assert len(self.b_locs) == 8
        assert len(self.s_locs) == 24
        assert len(self.r_locs) == 24

    def sim(self, a, b, s, fault=None):
        shape = a.shape
        a, b, s = a.astype(np.int8), b.astype(np.int8), s.astype(np.int32)
        nbytes = cdiv(len(a), 8)

        self.s[0, self.a_locs, 0, :nbytes] = np.packbits(unpackbits(a).T, axis=-1)
        self.s[0, self.b_locs, 0, :nbytes] = np.packbits(unpackbits(b).T, axis=-1)
        self.s[0, self.s_locs, 0, :nbytes] = np.packbits(unpackbits(s).T[:24], axis=-1)

        self.s_to_c()

        def make_fault_injector(fault_site, fault_value):
            def fi(line, data):
                if line == fault_site:
                    data[0] = 255 if fault_value else 0
            return fi

        self.c_prop(inject_cb=make_fault_injector(*fault) if fault else None)
        self.c_to_s()
        r_bits = np.unpackbits(self.s[1, self.r_locs, 0, :nbytes], axis=-1).T
        return packbits(r_bits, dtype=np.int32)[:len(a)].reshape(shape)

def collect_fault_sites(circuit):
    '''fault site: output of a gate.
    TODO: Proper stuck-at fault collapsing'''
    fault_sites = set()
    for n in circuit.topological_order():
        if n.kind == '__fork__': continue
        fault_sites |= set(n.outs)
    return [line.index for line in fault_sites if line is not None]


@dataclass
class FaultStat:
    fault_site: int
    fault_value: int
    failed_tests: int
    rmse: float


if __name__ == '__main__':
    netlist_path = Path('synthesized', 'pe.saed90.v')
    log.info(f"Loading {netlist_path}")
    circuit = verilog.load(netlist_path, tlib=SAED90)
    log.info(f'Lines: {len(circuit.lines)} Cells: {len(circuit.cells)}')
    fault_sites = collect_fault_sites(circuit)
    log.info(f'FaultSites: {len(fault_sites)}')
    circuit.resolve_tlib_cells(SAED90)
    fma = FMAlogic(circuit, sims=1024)
    log.info(f'Tests: {fma.sims}')

    rng = np.random.default_rng(42)
    a = rng.integers(-128, 127, fma.sims)
    b = rng.integers(-128, 127, fma.sims)
    s = rng.integers(-2**20, 2**20, fma.sims)

    golden = fma.sim(a, b, s)
    log.info(f"golden simulation finished {golden[:5]}")

    faults_stats: list[FaultStat] = []
    log.info("fault injection")
    for fault_site in tqdm(fault_sites):
        for fault_value in [0, 1]:
            syndrome = fma.sim(a, b, s, fault=(fault_site, fault_value))
            failed_tests = np.sum(syndrome != golden)
            rmse = np.sqrt(np.mean((syndrome.astype(float) - golden.astype(float)) ** 2))
            faults_stats.append(FaultStat(fault_site, fault_value, failed_tests, rmse))

    faults_stats.sort(key=lambda x: x.rmse, reverse=True)
    log.info(f'Fault coverage: {sum([fstat.failed_tests > 0 for fstat in faults_stats])/len(faults_stats)*100:.2f}%')
    log.info("Faults with Top 10 RMSE:")
    for fstat in faults_stats[:10]:
        log.info(f"  {fstat.fault_site} SA-{fstat.fault_value} failed_tests {fstat.failed_tests} rmse {fstat.rmse:.4f}")
