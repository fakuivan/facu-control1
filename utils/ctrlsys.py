from .math import ratpoly_coeffs
from sympy import Dummy, laplace_transform
from scipy import signal
import numpy as np

def expr_to_sys(expr, var, freq_space = False):
    s = var if freq_space else Dummy('s')
    tf = expr if freq_space else laplace_transform(expr, var, s, noconds=True)
    n, d = ratpoly_coeffs(tf*s, s)
    return signal.TransferFunction([*map(float, n)], [*map(float, d)])

def add_bode(sys, ax_mag, ax_phase, label=None, w=None):
    w, mag, phase = signal.bode(sys) if w is None else \
        signal.bode(sys, w)
    ax_mag.semilogx(w, mag, label=label)
    ax_phase.semilogx(w, phase, "--", label=label)

def scaled_step(prop, sys, *args, **kwargs):
    t, r = sys.step(*args, **kwargs)
    return t, r*prop

def add_step(sys, output_axis, t=None, input_axis=None, ampl=1, i_label=None, o_label=None):
    t, output = sys.step() if t is None else sys.step(T=t)
    output *= ampl
    input = np.ones(len(t))*ampl
    if input_axis is not None:
        input_axis.plot(t, input, label=i_label)
    output_axis.plot(t, output, label=o_label)
    return t, (input, output)