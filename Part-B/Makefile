SDK_INSTALL_PATH :=  /usr/local/cuda#/cm/shared/apps/cuda11.2/toolkit/11.2.2
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#######################################################################
clean:
	rm -f $(EXECS) *.o
#######################################################################
timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)
#######################################################################
q1.o: q1.cpp
	${NVCC} $< -c -o $@ $(OPTIONS)
#######################################################################
q1 : q1.o timer.o 
	${NVCC} $< q1.o -o $@ $(LIB) timer.o $(OPTIONS)

