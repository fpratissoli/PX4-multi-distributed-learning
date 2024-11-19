import numpy as np
import utilities as utils
import matplotlib.pyplot as plt
from robot import Robot
import time
import socket
import threading
import json

# Generate data
np.random.seed(0)

area_size = 200
x_inf, y_inf = 0, 0
x_sup, y_sup = area_size, area_size
BBOX = [x_inf, y_inf, x_sup, y_sup]
d_field_ = 5
x1_ = np.arange(x_inf, x_sup, d_field_)
x2_ = np.arange(y_inf, y_sup, d_field_)
_X1, _X2 = np.meshgrid(x1_, x2_)
mesh = np.vstack([_X1.ravel(), _X2.ravel()]).T

# Generate ra05ndom means
peaks = 4 # np.random.randint(1, 10)
means = np.random.uniform(low=0, high=area_size, size=(peaks, 2))
sigma = 30
Z = utils.gmm_pdf_array(mesh[:, 0], mesh[:, 1], sigma, means, flag_normalize=False)
Z = Z.reshape(len(x1_), len(x2_))
field = Z
# np.save("field.npy", field)

""" Robots parameters """
ROB_NUM = 6
CAMERA_BOX = 20
CAMERA_SAMPLES = 5

_area_to_cover = (x_sup * y_sup) * 2.0
RANGE = 2 * np.sqrt((_area_to_cover / ROB_NUM) / np.pi)

K_GAIN = 5
D_t = 0.1

robots = np.empty(ROB_NUM, dtype=object)
safety_dist = 5 # Safety distance to the borders

# Grid close initial positions
init_poses = np.array([[10, 10], [10, 30], [30, 10], [30, 30], [10, 50], [30, 50]])
for r in np.arange(ROB_NUM):
    # x1, x2 = np.random.uniform(0 + safety_dist, (x_sup - safety_dist)), np.random.uniform(0 + safety_dist, (y_sup - safety_dist))
    x1, x2 = init_poses[r]
    rob = Robot(total_robots=ROB_NUM,
                id=r,
                x1_init=x1,
                x2_init=x2,
                x1Vals=x1_,
                x2Vals=x2_,
                sensing_range=RANGE,
                sensor_noise=0.2,
                bbox=BBOX,
                mesh=mesh,
                field_delta=d_field_)
    robots[r] = rob

""" Figures """
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
#fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
#plt.rcParams["pdf.fonttype"] = 42

PERIOD = 300
index_init = 600
index_end = 1400
step = (index_end - index_init) // PERIOD
index = index_init

""" Hystories """
robotHistory = np.empty((ROB_NUM, 2, PERIOD)) # History of the robots' positions
droneHistory = np.empty((ROB_NUM, 2, PERIOD*20)) # History of the drones' positions
real_t=0

""" Network parameters """
A = np.zeros((ROB_NUM, ROB_NUM)) # Adjacency matrix

""" DEC-apx-GP """
s_end_DEC_gapx = 100
rho = 500
ki = 5000
TOL_ADMM = 1e-3

""" DEC-PoE """
beta = 1 / ROB_NUM
s_end_DAC = 100

""" global variables for threads"""
agents_pos_received = False
do_plot = False
agents_pos_temp = np.empty((ROB_NUM, 2))


def send_coordinates(coords):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 8888))
        s.sendall(json.dumps(coords).encode())

def handle_client(client_socket):
    while True:
        global real_t
        data = client_socket.recv(1024)
        if not data:
            break
        positions = json.loads(data.decode())
        print("Received drone positions: ------------------------------------------------------")
        for pos in positions:
            print(f"Drone {pos['drone_id']}: Lat {pos['latitude']}, Lon {pos['longitude']}, Alt {pos['altitude']}")
            #robots[pos['drone_id']-1].position = np.array([pos['latitude'], pos['longitude']])
            agents_pos_temp[pos['drone_id']-1] = np.array([pos['latitude'], pos['longitude']])
            droneHistory[pos['drone_id']-1, :, real_t] = np.array([pos['latitude'], pos['longitude']])

        real_t += 1
        global agents_pos_received
        agents_pos_received = True
    client_socket.close()

def perform_other_operation():
    """
    Main loop
    """
    while agents_pos_received is False:
        print("Waiting for agents positions...")
        time.sleep(1)

    for i, robot in enumerate(robots):
        robot.position = agents_pos_temp[i]

    for t in np.arange(0, PERIOD):
        print(f"\n=== Step: {t} ===")

        if t == 0:
            first = True
        else:
            first = False

        utils.sense_neighbors(robots)
        utils.update_A(robots, A)
        utils.find_groups(robots, A)
        groups = [robot.group for robot in robots]

        degrees = np.sum(A, axis=1)
        max_degree = np.max(degrees)

        eps = 1 / (max_degree)
        eps = eps / 2

        for i, robot in enumerate(robots):
            robot.time = t
            robot.compute_voronoi()

            # Take 5 random points from the robot sensing area
            points = np.random.uniform(robot.position - CAMERA_BOX, robot.position + CAMERA_BOX, (int(CAMERA_SAMPLES), 2))
            # Take the samples in a grid from the camera box
            # points = np.array([[x, y] for x in np.linspace(robot.position[0] - CAMERA_BOX, robot.position[0] + CAMERA_BOX, CAMERA_SAMPLES) for y in np.linspace(robot.position[1] - CAMERA_BOX, robot.position[1] + CAMERA_BOX, CAMERA_SAMPLES)])
            points = np.clip(points, [0, 0], [x_sup, y_sup])
            y_values = utils.gmm_pdf_array(points[:, 0], points[:, 1], sigma, means, flag_normalize=False) + robot.sensor_noise * np.random.randn(len(points))
            # Force the samples to be inside the field
            # y_values = utils.evaluate_points_in_field(field, points, method='linear') + robot.sensor_noise * np.random.randn(len(points))
            
            robot.sense(points, y_values, first=first)
            robot.update_dataset() # Update the dataset with the new observation

        for robot in robots:
            # robot.update_dataset()
            print(f"Robot {robot.id} has {robot.observations.shape[0]} observations")

        for robot in robots:
            robot.filter_dataset()
            print(f"Robot {robot.id} has {robot.dataset.shape[0]} filtered observations")

        # DEC_gapx_time_vec = []
        # DAC_time_vec = []
        for group in np.unique(groups):
            print(f"*** Processing group: {group} ***")
            # Sort the robots by id in the group
            group_robots = [robot for robot in robots if robot.group == group]
            group_robots = sorted(group_robots, key=lambda x: x.id)
            DEC_gapx_time, DAC_time = utils.process_group(group_robots,s_end_DEC_gapx, s_end_DAC, rho, ki, beta, eps, x1_, x2_, ROB_NUM)
            # DEC_gapx_time_vec.append(DEC_gapx_time)
            # DAC_time_vec.append(DAC_time)

        # Mean computation time
        # DEC_gapx_time = np.mean(DEC_gapx_time_vec)
        # DAC_time = np.mean(DAC_time_vec)

        # DEC_gapx_time_hist[t] = DEC_gapx_time
        # DAC_time_hist[t] = DAC_time

        for i, robot in enumerate(robots):
            robot.compute_centroid()
            robotHistory[i, :, t] = robot.position

        # utils.plot_dataset(fig, t, PERIOD, BBOX, field, ax1, x1_, x2_, _X1, _X2, robots, A)

        # Move the robots
        for robot in robots:
            x1, x2 = robot.position + (-K_GAIN * (robot.position - robot.centroid) * D_t)
            robot.move(x1, x2)

        coords = [[robot.position[0], robot.position[1], 20] for robot in robots]
        send_coordinates(coords)
        print(f"------ Sent coordinates: {coords}")

        global do_plot
        do_plot = True

        # wait the drones reach the new positions
        time.sleep(3)

def perform_plotting():
    plt.ion()
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    #plt.rcParams["pdf.fonttype"] = 42
    while True:
        if do_plot is True:
            utils.plot_dataset(fig, PERIOD, BBOX, field, ax1, x1_, x2_, _X1, _X2, robots, A)

def perform_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 8889))
    server.listen(5)
    print("Position receiver serving on ('localhost', 8889)")

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        # Create a new thread for each client connection
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

def main():
    # server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.bind(('localhost', 8889))
    # server.listen(5)
    # print("Position receiver serving on ('localhost', 8889)")

    # # Start the server in a separate thread
    server_thread = threading.Thread(target=perform_server, daemon=True)
    server_thread.start()

    # Start background operations in a separate thread
    background_thread = threading.Thread(target=perform_other_operation, daemon=True)
    background_thread.start()

    # Start plotting in the main thread
    perform_plotting()

    # while True:
    #     client_socket, addr = server.accept()
    #     print(f"Accepted connection from {addr}")
    #     # Create a new thread for each client connection
    #     client_thread = threading.Thread(target=handle_client, args=(client_socket,))
    #     client_thread.start()

if __name__ == "__main__":
    main()