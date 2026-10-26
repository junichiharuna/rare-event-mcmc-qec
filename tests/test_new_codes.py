import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from css_code import (
    make_asymmetric_toric_code,
    make_surface_code,
    make_bb_code_72,
    make_bb_code_144,
    make_bposd_decoder,
    make_depolarizing_bposd_decoder,
)


def _check_css(code):
    assert np.all((code.hx @ code.hz.T) % 2 == 0), "CSS: HX @ HZ^T ≠ 0"
    assert np.all((code.hz @ code.lx.T) % 2 == 0), "HZ @ LX^T ≠ 0"
    assert np.all((code.hx @ code.lz.T) % 2 == 0), "HX @ LZ^T ≠ 0"
    for arr in (code.hx, code.hz, code.lx, code.lz):
        assert arr.dtype == np.int8


class TestAsymmetricToricCode:
    def test_shape_3_6(self):
        code = make_asymmetric_toric_code(3, 6)
        assert code.n == 2 * 3 * 6  # 36

    def test_shape_5_10(self):
        code = make_asymmetric_toric_code(5, 10)
        assert code.n == 2 * 5 * 10  # 100

    def test_css_property(self):
        _check_css(make_asymmetric_toric_code(3, 6))

    def test_k_equals_2(self):
        code = make_asymmetric_toric_code(3, 6)
        assert code.lx.shape[0] == 2


class TestSurfaceCode:
    def test_n_d3(self):
        code = make_surface_code(3)
        assert code.n == 13  # 2*9 - 6 + 1

    def test_n_d5(self):
        code = make_surface_code(5)
        assert code.n == 41  # 2*25 - 10 + 1

    def test_css_property(self):
        _check_css(make_surface_code(3))

    def test_k_equals_1(self):
        code = make_surface_code(3)
        assert code.lx.shape[0] == 1


class TestBBCode72:
    def test_n_and_k(self):
        code = make_bb_code_72()
        assert code.n == 72
        assert code.lx.shape[0] == 12

    def test_css_property(self):
        _check_css(make_bb_code_72())

    def test_lx_lz_valid(self):
        code = make_bb_code_72()
        # LX @ LZ^T should have full rank (symplectic)
        comm = (code.lx @ code.lz.T) % 2
        assert np.linalg.matrix_rank(comm.astype(float)) == 12


class TestBBCode144:
    def test_n_and_k(self):
        code = make_bb_code_144()
        assert code.n == 144
        assert code.lx.shape[0] == 12

    def test_css_property(self):
        _check_css(make_bb_code_144())


class TestBPOSDDecoder:
    def test_zero_syndrome_zero_correction(self):
        code = make_bb_code_72()
        dec = make_bposd_decoder(code, error_rate=1e-2)
        syn = np.zeros(code.hz.shape[0], dtype=np.uint8)
        corr = dec.decode_syndrome(syn)
        assert corr.shape == (code.n,)
        assert np.all(corr == 0)

    def test_not_faulty_trivial(self):
        code = make_bb_code_72()
        dec = make_bposd_decoder(code, error_rate=1e-2)
        zeros = np.zeros(code.n, dtype=np.int8)
        assert dec.is_faulty(zeros, zeros) is False

    def test_faulty_logical_error(self):
        code = make_bb_code_72()
        dec = make_bposd_decoder(code, error_rate=1e-2)
        lx_err = code.lx[0].astype(np.int8)
        zeros = np.zeros(code.n, dtype=np.int8)
        assert dec.is_faulty(lx_err, zeros) is True


class TestDepolarizingBPOSDDecoder:
    def test_zero_syndromes(self):
        code = make_bb_code_72()
        dec = make_depolarizing_bposd_decoder(code, error_rate=1e-2)
        zero_sz = np.zeros(code.hz.shape[0], dtype=np.uint8)
        zero_sx = np.zeros(code.hx.shape[0], dtype=np.uint8)
        c_x = dec.decode_x(zero_sz)
        c_z = dec.decode_z(zero_sx)
        assert np.all(c_x == 0)
        assert np.all(c_z == 0)

    def test_not_faulty_trivial(self):
        code = make_bb_code_72()
        dec = make_depolarizing_bposd_decoder(code, error_rate=1e-2)
        zeros = np.zeros(code.n, dtype=np.int8)
        assert dec.is_faulty(zeros, zeros, zeros, zeros) is False

    def test_faulty_logical_x(self):
        code = make_bb_code_72()
        dec = make_depolarizing_bposd_decoder(code, error_rate=1e-2)
        lx_err = code.lx[0].astype(np.int8)
        zeros = np.zeros(code.n, dtype=np.int8)
        assert dec.is_faulty(lx_err, zeros, zeros, zeros) is True

    def test_faulty_logical_z(self):
        code = make_bb_code_72()
        dec = make_depolarizing_bposd_decoder(code, error_rate=1e-2)
        lz_err = code.lz[0].astype(np.int8)
        zeros = np.zeros(code.n, dtype=np.int8)
        assert dec.is_faulty(zeros, lz_err, zeros, zeros) is True
