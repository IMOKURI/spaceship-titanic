from src.feature_store import Store


def test_feature_store():
    store = Store.empty()

    assert type(store) == Store
