import numpy as np
from matplotlib import pyplot as plt

# extract the test results
def extractResult(filename):
    data = np.loadtxt(filename)
    xlist = data[0]
    ylist = data[1]
    dict = {"x": xlist, "y": ylist}
    return dict

# CPU versus GPU
plot1_cpu = extractResult('results_cpu.txt')
scaled_cpu = 1000*plot1_cpu["y"]
plot1_gpu = extractResult('results_gpu.txt')
fig1 = plt.figure(1)
plt.plot(plot1_cpu["x"], scaled_cpu, marker="o", color="red", label='CPU')
plt.plot(plot1_gpu["x"], plot1_gpu["y"], marker="o", color="blue", label='GPU')
plt.title("Runtime of CPU versus GPU Implementation on Texture Synthesis")
plt.xlabel("Output Size")
plt.ylabel("Inclusive Runtime (ms)")
plt.legend(loc="best")
plt.show()

# GPU scaling analysis on output size
plot2 = extractResult('results_gpu_outputsize.txt')
fig2 = plt.figure(2)
plt.plot(plot2["x"], plot2["y"], marker="o", color="black",label='GPU')
plt.title("Scaling Analysis of GPU Runtime versus Output Texture Size")
plt.xlabel("Output Texture Size")
plt.ylabel("Inclusive Runtime (ms)")
plt.legend(loc="best")
plt.show()

# GPU scaling analysis on neighborhood size
plot3 = extractResult('results_gpu_neighsize.txt')
fig3 = plt.figure(3)
plt.plot(plot3["x"], plot3["y"], marker="o", color="black",label='GPU')
plt.title("Scaling Analysis of GPU Runtime versus Neighborhood Size")
plt.xlabel("Neighborhood Size")
plt.ylabel("Inclusive Runtime (ms)")
plt.legend(loc="best")
plt.show()

# GPU scaling analysis on iterations
plot4 = extractResult('results_gpu_iterations.txt')
fig4 = plt.figure(4)
plt.plot(plot4["x"], plot4["y"], marker="o", color="black",label='GPU')
plt.title("Scaling Analysis of GPU Runtime versus Iterations per Level")
plt.xlabel("Iterations per Level")
plt.ylabel("Inclusive Runtime (ms)")
plt.legend(loc="best")
plt.show()
