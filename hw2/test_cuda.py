from numba import cuda

def test_cuda_available():
    print(cuda.is_available())
    assert cuda.is_available() == True