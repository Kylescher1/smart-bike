import csv
import matplotlib.pyplot as plt

def read_csv(filename):
    time_data, ax, ay, az, gx, gy, gz = [], [], [], [], [], [], []

    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_data.append(float(row["Time(s)"]))
            ax.append(float(row["Ax(g)"]))
            ay.append(float(row["Ay(g)"]))
            az.append(float(row["Az(g)"]))
            gx.append(float(row["Gx(°/s)"]))
            gy.append(float(row["Gy(°/s)"]))
            gz.append(float(row["Gz(°/s)"]))

    return time_data, ax, ay, az, gx, gy, gz


def plot_data(time_data, ax, ay, az, gx, gy, gz):
    plt.style.use('ggplot')
    fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_acc.plot(time_data, ax, label="Ax")
    ax_acc.plot(time_data, ay, label="Ay")
    ax_acc.plot(time_data, az, label="Az")
    ax_acc.set_title("Accelerometer (g)")
    ax_acc.legend(loc="upper right")

    ax_gyro.plot(time_data, gx, label="Gx")
    ax_gyro.plot(time_data, gy, label="Gy")
    ax_gyro.plot(time_data, gz, label="Gz")
    ax_gyro.set_title("Gyroscope (°/s)")
    ax_gyro.legend(loc="upper right")

    ax_gyro.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = "sensor_log.csv"  # change if needed
    time_data, ax, ay, az, gx, gy, gz = read_csv(filename)
    plot_data(time_data, ax, ay, az, gx, gy, gz)
