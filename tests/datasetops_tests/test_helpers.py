from datasetops.helpers import documents, documented, parameters, signature


def test_documents():
    class ClassA:
        def func(self):
            return 42

    assert ClassA.func.__doc__ is None

    class ClassB:
        @documents(ClassA)
        def func(self):
            """Returns 42"""
            return 42

    assert ClassB.func.__doc__ == "Returns 42"
    assert ClassA.func.__doc__ == "Returns 42"


def test_documented():
    class ClassA:
        def func(self):
            """Returns 42"""
            return 42

    class ClassB:
        @documented(ClassA)
        def func(self):
            return 42

    assert ClassA.func.__doc__ == "Returns 42"
    assert ClassB.func.__doc__ == "Returns 42"


def test_parameters():
    def myfunc(required_param, other_param=42):
        local_var = 1
        return required_param + other_param + local_var

    assert parameters(myfunc) == {
        "required_param": None,
        "other_param": 42,
    }

    assert parameters(lambda x: x) == {"x": None}
    assert parameters(lambda: 42) == {}


def test_signature():
    def myfunc(required_param, other_param=42):
        local_var = 1
        return required_param + other_param + local_var

    assert signature(myfunc) == "myfunc(required_param=None, other_param=42)"

    assert signature(lambda x: x) == "<lambda>(x=None)"
    assert signature(lambda: 42) == "<lambda>()"
