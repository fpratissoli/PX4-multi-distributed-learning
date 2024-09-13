import numpy as np
import utilities as utils
import matplotlib.pyplot as plt
from robot import Robot
import json
import time
import socket
import threading

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

# Generate random means
peaks = 4 # np.random.randint(1, 10)
means = np.random.uniform(low=0, high=area_size, size=(peaks, 2))
sigma = 30
Z = utils.gmm_pdf_array(mesh[:, 0], mesh[:, 1], sigma, means, flag_normalize=False)
Z = Z.reshape(len(x1_), len(x2_))
field = Z

""" Robots parameters """
ROB_NUM = 6
CAMERA_BOX = 30
CAMERA_SAMPLES = 20

_area_to_cover = (x_sup * y_sup) * 2.0
RANGE = 2 * np.sqrt((_area_to_cover / ROB_NUM) / np.pi)

K_GAIN = 3
D_t = 0.1

robots = np.empty(ROB_NUM, dtype=object)
safety_dist = 5 # Safety distance to the borders
for r in np.arange(ROB_NUM):
    x1, x2 = np.random.uniform(0 + safety_dist, (x_sup - safety_dist)), np.random.uniform(0 + safety_dist, (y_sup - safety_dist))
    rob = Robot(total_robots=ROB_NUM,
                id=r,
                x1_init=x1,
                x2_init=x2,
                x1Vals=x1_,
                x2Vals=x2_,
                sensing_range=RANGE,
                sensor_noise=0.1,
                bbox=BBOX,
                mesh=mesh,
                field_delta=d_field_)
    robots[r] = rob

PERIOD = 300

""" Network parameters """
A = np.zeros((ROB_NUM, ROB_NUM)) # Adjacency matrix

""" DEC-apx-GP Parameters """
s_end_DEC_gapx = 100
rho = 500
ki = 5000
TOL_ADMM = 1e-3

""" DEC-PoE Parameters """
beta = 1 / ROB_NUM
s_end_DAC = 100

agents_pos_received = False
do_plot = False

def send_coordinates(coords):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 8888))
        s.sendall(json.dumps(coords).encode())

def handle_client(client_socket):
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        positions = json.loads(data.decode())
        print("Received drone positions: ------------------------------------------------------")
        for pos in positions:
            print(f"Drone {pos['drone_id']}: Lat {pos['latitude']}, Lon {pos['longitude']}, Alt {pos['altitude']}")
            robots[pos['drone_id']-1].position = np.array([pos['latitude'], pos['longitude']])

        global agents_pos_received
        agents_pos_received = True
    client_socket.close()

def perform_other_operation():
    """
    Main loop
    """
    start_time = time.time()
    while agents_pos_received is False:
        print("Waiting for agents positions...")
        time.sleep(1)

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

        eps = 1 / (max_degree) - 0.01 # To avoid a path network bug

        for i, robot in enumerate(robots):
            robot.time = t
            robot.compute_voronoi()

            # Take random points from the robot sensing area
            points = np.random.uniform(robot.position - CAMERA_BOX, robot.position + CAMERA_BOX, (int(CAMERA_SAMPLES), 2))
            y_values = utils.gmm_pdf_array(points[:, 0], points[:, 1], sigma, means, flag_normalize=False) + robot.sensor_noise * np.random.randn(len(points))
            # y_values = utils.evaluate_points_in_field(field, points, method='linear') + robot.sensor_noise * np.random.randn(len(points))
            
            robot.sense(points, y_values, first=first)
            robot.update_dataset() # Update the dataset with the new observation

        for robot in robots:
            # robot.update_dataset()
            print(f"Robot {robot.id} has {robot.observations.shape[0]} observations")

        for robot in robots:
            # print(f"Robot {robot.id} has {robot.observations.shape[0]} observations")
            robot.filter_dataset()
            print(f"Robot {robot.id} has {robot.dataset.shape[0]} filtered observations")

        for group in np.unique(groups):
            print(f"*** Processing group: {group} ***")
            group_robots = [robot for robot in robots if robot.group == group]
            group_robots = sorted(group_robots, key=lambda x: x.id)
            utils.process_group(group_robots,s_end_DEC_gapx, s_end_DAC, rho, ki, beta, eps, x1_, x2_, ROB_NUM)
        
        for i, robot in enumerate(robots):
            robot.compute_centroid()

        # Move the robots
        # for robot in robots:
        #     x1, x2 = robot.position + (-K_GAIN * (robot.position - robot.centroid) * D_t)
        #     robot.move(x1, x2)

        coords = [[robot.centroid[0], robot.centroid[1], 20] for robot in robots]
        passed_time = time.time() - start_time
        if passed_time >= 8: # Send coordinates every x seconds
            send_coordinates(coords)
            start_time = time.time()
            print(f"------ Sent coordinates: {coords}")
        #time.sleep(1)
        global do_plot
        do_plot = True
        #utils.plot_dataset(d_field_, fig, t, PERIOD, BBOX, field, ax1, ax2, ax3, x1_, x2_, _X1, _X2, robots, A)

def perform_plotting():
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    while True:
        if do_plot is True:
            utils.plot_dataset(d_field_, fig, PERIOD, BBOX, field, ax1, ax2, ax3, x1_, x2_, _X1, _X2, robots, A)

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
    # Start the server in a separate thread
    server_thread = threading.Thread(target=perform_server, daemon=True)
    server_thread.start()

    # Start background operations in a separate thread
    background_thread = threading.Thread(target=perform_other_operation, daemon=True)
    background_thread.start()

    # Start plotting in the main thread
    perform_plotting()

if __name__ == "__main__":
    main()