SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64

OPTIONS   :=  -O3 
#######################################################################
EXECS :=  c1  c2 c3
all:$(EXECS)
#######################################################################
clean:
	rm -f $(EXECS) *.o
#######################################################################
timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)
#######################################################################
c1.o: c1.cu
	${NVCC} $< -c -o $@ $(OPTIONS)
#######################################################################
c1 : c1.o timer.o 
	${NVCC} $^ -o $@ $(LIB) $(OPTIONS)

#######################################################################
c2.o: c2.cu
	${NVCC} $< -c -o $@ $(OPTIONS)
#######################################################################
c2 : c2.o timer.o 
	${NVCC} $^ -o $@ $(LIB) $(OPTIONS) 

#######################################################################
c3.o: c3.cu #### -lcudnn to link
	${NVCC} $< -c -o $@ $(OPTIONS) -lcudnn 
#######################################################################
c3 : c3.o timer.o 
	${NVCC} $^ -o $@ $(LIB) $(OPTIONS) -lcudnn