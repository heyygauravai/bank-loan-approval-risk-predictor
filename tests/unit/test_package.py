from importlib import import_module


def test_package_can_be_imported() -> None:
    package = import_module("loan_approval_prediction")

    assert package.__name__ == "loan_approval_prediction"
