import numpy as np
import matplotlib.pyplot as plt

# Values of K (in millions)
K_values = [1, 5, 10, 50, 100]

# Time to execute the program on GPU for Step 3
gpu_times_step3_1 = [0.000007, 0.000023, 0.000028, 0.000026, 0.000031]
gpu_times_step3_2 = [0.000004, 0.000004, 0.000004, 0.000003, 0.000004]
gpu_times_step3_3 = [0.000003, 0.000005, 0.000006, 0.000005, 0.000005]

# Time to execute the program on GPU for Step 2
gpu_times_step2_1 = [0.000011, 0.000027, 0.000033, 0.000030, 0.000035]
gpu_times_step2_2 = [0.000003, 0.000006, 0.000004, 0.000004, 0.000003]
gpu_times_step2_3 = [0.000004, 0.000004, 0.000017, 0.000006, 0.000000]

# Combining into single arrays for Step 2 and Step 3
gpu_times_step3 = np.array([gpu_times_step3_1, gpu_times_step3_2, gpu_times_step3_3])
gpu_times_step2 = np.array([gpu_times_step2_1, gpu_times_step2_2, gpu_times_step2_3])

# Plotting Step 2 and Step 3
plt.figure(figsize=(10, 6))

# Plotting Step 2
for i in range(len(gpu_times_step2)):
    plt.plot(K_values, gpu_times_step2[i], label=f' Q2 -> GPU (Step 2) - Run {i+1}', marker='o')

# Plotting Step 3
for i in range(len(gpu_times_step3)):
    plt.plot(K_values, gpu_times_step3[i], label=f' Q3 -> GPU (Step 3) - Run {i+1}', marker='s')

plt.xlabel('Value of K (Million)')
plt.ylabel('Time (sec)')
plt.title('Execution Time Comparison')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()
