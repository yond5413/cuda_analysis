#include <stdio.h>
#include <cstdlib> 
#include <cstring> 
#include "timer.h"

// Function to add elements of two arrays
void addArrays(int* arr1, int* arr2, int size) {
    for (int i = 0; i < size; ++i) {
        arr1[i] += arr2[i];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
       printf("Usage: %s <K>\n", argv[0]);
       printf("Need to add input for length of array");
        return 1;
    }

    int K = atoi(argv[1]);
    int million = 1000000;
    int size = K * million;

   
    // Allocate memory for arrays
    int* array1 = (int*)malloc(size * sizeof(int));//new int[size];
    int* array2 = (int*)malloc(size * sizeof(int));//new int[size];

    // Initialize arrays with random values
    for (int i = 0; i < size; ++i) {
        array1[i] = 1; //rand() % 100;
        array2[i] = 1;//rand() % 100;
    }

    // Profile and get the time to execute
    initialize_timer();
    start_timer();
    // Add arrays
    addArrays(array1, array2, size);

    stop_timer();
    double duration = elapsed_time();
    
   printf("Time taken for adding arrays with K = %d million elements: %lf seconds\n", K, duration);


    // Free memory
    free(array1);//delete[] array1;
    free(array2);//delete[] array2;

    return 0;
}