import pytest
import flowdyn.mesh
import flowdyn.field
import flowdyn.integration
import flowdyn.modelphy.base as modelbase

def test_importpytest():
	assert 1

def test_runpytest():
	assert 1

def test_base_model_repr():
    model = modelbase.model(name="test", neq=2)
    assert repr(model) == "model: test\nnb eq: 2"

def test_base_model_abstract_operations():
    model = modelbase.model()
    with pytest.raises(NotImplementedError):
        model.cons2prim([])
    with pytest.raises(ValueError, match="unknown variable"):
        model.nameddata("missing", [])
