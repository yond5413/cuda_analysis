Hello welcome to some of my analysis for on cuda below I will discuss how to run each of the experiments in the directories here

&nbsp; In the Part-A directory in order to test each of the programs we have to use the make command in the cli to compile and execute our programs, and the follwing excutables will be generated  vecadd00, vecadd01, matmult00, matmult01.
- Each of these files can be excuted by entering the commnad 
        - ./foo <integer>   
        - In the vecadd case this integer will be the number of elements values added per thread
        - in the matmult case it correspond to the number of blocks which directly influences the data dimension of the matrix 
    
&nbsp; To create executables for the assignment there is a makefile available in the Part-B directory
all that is needed is to enter make in the comamand line, and *.o and other executables will be prepeared
&nbsp; Once that is done 3 executables will be in the directory ql,q2, and q3 which represent each part of the assignment. 
&nbsp; To execute q1 enter in cli
    &nbsp;&nbsp; ./q1 <k-value>
&nbsp; To execute q2 enter in cli
    &nbsp;&nbsp; ./q2 <scenario> <k-value>
&nbsp; To execute q1 enter in cli
    &nbsp;&nbsp; ./q2 <scenario> <k-value>
In this case scenario represents one of the three cases 
-  Using one block with 1 thread
- Using one block with 256 threads
- Using multiple blocks with 256 threads per block with the total number of threads
 across all the blocks equal to the size of arrays.

&nbsp; In the Part-C directory none of the executables generate by have command line arguements, so in this case just enter the file name ,/name after using the make command 