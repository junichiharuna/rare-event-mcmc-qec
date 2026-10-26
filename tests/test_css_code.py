import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from css_code import CSSCode, make_toric_code, make_mwpm_decoder


def test_make_toric_code_d3_shape():
    code = make_toric_code(3)
    assert isinstance(code, CSSCode)
    assert code.n == 18  # 2 * d^2 = 18
    assert code.hx.shape == (9, 18)
    assert code.hz.shape == (9, 18)
    assert code.lx.shape[1] == 18
    assert code.lz.shape[1] == 18


def test_make_toric_code_d3_dtype():
    code = make_toric_code(3)
    assert code.hx.dtype == np.int8
    assert code.hz.dtype == np.int8
    assert code.lx.dtype == np.int8
    assert code.lz.dtype == np.int8


def test_css_property_hx_hz():
    # CSS: hz @ hx^T = 0 (mod 2) — StabFlip syndrome-preservation
    code = make_toric_code(3)
    assert np.all((code.hz @ code.hx.T) % 2 == 0)


def test_css_property_hz_lx():
    # CSS: hz @ lx^T = 0 (mod 2) — LogicFlip syndrome-preservation
    code = make_toric_code(3)
    assert np.all((code.hz @ code.lx.T) % 2 == 0)


def test_mwpm_decoder_zero_syndrome():
    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    zero_syn = np.zeros(code.hz.shape[0], dtype=np.uint8)
    correction = dec.decode_syndrome(zero_syn)
    assert isinstance(correction, np.ndarray)
    assert len(correction) == code.n
    assert np.all(correction == 0)


def test_mwpm_decoder_is_faulty_trivial():
    # zero error + zero correction → not faulty
    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    zero_error = np.zeros(code.n, dtype=np.int8)
    zero_corr = np.zeros(code.n, dtype=np.int8)
    assert dec.is_faulty(zero_error, zero_corr) == False


def test_mwpm_decoder_is_faulty_logical():
    # lx[0] error, zero correction → residual = lx[0], faulty
    code = make_toric_code(3)
    dec = make_mwpm_decoder(code)
    logical_error = code.lx[0].astype(np.int8)
    zero_corr = np.zeros(code.n, dtype=np.int8)
    assert dec.is_faulty(logical_error, zero_corr) == True


def test_depolarizing_decoder_zero_syndromes():
    from css_code import make_depolarizing_mwpm_decoder

    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    zero_sz = np.zeros(code.hz.shape[0], dtype=np.uint8)
    zero_sx = np.zeros(code.hx.shape[0], dtype=np.uint8)
    c_x = dec.decode_x(zero_sz)
    c_z = dec.decode_z(zero_sx)
    assert isinstance(c_x, np.ndarray) and len(c_x) == code.n
    assert isinstance(c_z, np.ndarray) and len(c_z) == code.n
    assert np.all(c_x == 0)
    assert np.all(c_z == 0)


def test_depolarizing_is_faulty_trivial():
    from css_code import make_depolarizing_mwpm_decoder

    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    zeros = np.zeros(code.n, dtype=np.int8)
    assert dec.is_faulty(zeros, zeros, zeros, zeros) == False


def test_depolarizing_is_faulty_logical_x():
    from css_code import make_depolarizing_mwpm_decoder

    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    lx_row = code.lx[0].astype(np.int8)
    zeros = np.zeros(code.n, dtype=np.int8)
    # e_x = lx[0], c_x = zeros → residual_X = lx[0] → lz @ lx[0] ≠ 0 → faulty
    assert dec.is_faulty(lx_row, zeros, zeros, zeros) == True


def test_depolarizing_is_faulty_logical_z():
    from css_code import make_depolarizing_mwpm_decoder

    code = make_toric_code(3)
    dec = make_depolarizing_mwpm_decoder(code)
    lz_row = code.lz[0].astype(np.int8)
    zeros = np.zeros(code.n, dtype=np.int8)
    # e_z = lz[0], c_z = zeros → residual_Z = lz[0] → lx @ lz[0] ≠ 0 → faulty
    assert dec.is_faulty(zeros, lz_row, zeros, zeros) == True


def test_make_bp_decoder_is_faulty_branches():
    from css_code import make_bp_decoder

    code = make_toric_code(3)
    dec = make_bp_decoder(code, error_rate=0.05)

    n = code.n
    zero = np.zeros(n, dtype=np.int8)

    # (a) no error, no correction -> not faulty
    assert dec.is_faulty(zero, zero) is False

    # (b) X-logical as error, zero correction: syndrome 0 (converged), wrong coset -> faulty
    lx0 = code.lx[0].astype(np.int8)
    assert not (code.hz @ lx0 % 2).any()  # lx commutes with Z-stabilizers
    assert dec.is_faulty(lx0, zero) is True

    # (c) X-stabilizer as error, zero correction: syndrome 0, trivial coset -> not faulty
    hx0 = code.hx[0].astype(np.int8)
    assert dec.is_faulty(hx0, zero) is False

    # (d) correction that fails to reproduce syndrome -> faulty (non-convergence branch)
    e = np.zeros(n, dtype=np.int8)
    e[0] = 1
    assert (code.hz @ e % 2).any()  # single bit has a non-trivial syndrome
    assert dec.is_faulty(e, zero) is True


def test_make_bp_decoder_decode_zero_syndrome():
    from css_code import make_bp_decoder

    code = make_toric_code(3)
    dec = make_bp_decoder(code, error_rate=0.05)
    s = np.zeros(code.hz.shape[0], dtype=np.uint8)
    c = dec.decode_syndrome(s)
    assert c.shape == (code.n,)
    assert int(c.sum()) == 0
