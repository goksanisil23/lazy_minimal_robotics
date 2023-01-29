import matplotlib.pyplot as plt

estimated_states = []
true_states = []
measurements = []

with open("./build/out.txt", "r") as file:
    for line in file:
        elements = line.strip().split(" ")
        estimated_states.append((float(elements[0]), float(elements[1])))
        true_states.append((float(elements[2]), float(elements[3])))
        measurements.append((float(elements[4]), float(elements[5])))

x_estimated = [x for x, y in estimated_states]
y_estimated = [y for x, y in estimated_states]

x_true = [x for x, y in true_states]
y_true = [y for x, y in true_states]

x_meas = [x for x, y in measurements]
y_meas = [y for x, y in measurements]

plt.plot(x_estimated, y_estimated, "ro", label="Estimated States")
plt.plot(x_true, y_true, "b", label="True States")
plt.plot(x_meas, y_meas, "gx", label="Measurements")

plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Estimated States vs True States")
plt.legend()
plt.show()
