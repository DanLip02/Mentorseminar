import pycuda.autoinit
import pycuda.driver as drv
print("PyCUDA работает! Версия драйвера CUDA:", drv.get_driver_version())