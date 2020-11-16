CUCOMP  = nvcc
CUFLAGS = -arch=sm_70

INCLUDES  = -I$(OLCF_ESSL_ROOT)/include
LIBRARIES = -L$(CUDA_DIR)/lib64 -L$(OLCF_ESSL_ROOT)/lib64 -lcublas -lessl

cublas_dgemm_batched: cublas_dgemm_batched.o
	$(CUCOMP) $(CUFLAGS) $(LIBRARIES) cublas_dgemm_batched.o -o cublas_dgemm_batched

cublas_dgemm_batched.o: cublas_dgemm_batched.cu
	$(CUCOMP) $(CUFLAGS) $(INCLUDES) -c cublas_dgemm_batched.cu

.PHONY: clean

clean:
	rm -f cublas_dgemm_batched *.o
