"""
Tests for opticalib.core.data_classes module.
"""

from opticalib.core.data_classes import FlatData


def test_flatdata_import_from_dmutils():
    """FlatData should still be re-exported from dmutils."""
    from opticalib.dmutils import FlatData as ReExportedFlatData

    assert ReExportedFlatData is FlatData
