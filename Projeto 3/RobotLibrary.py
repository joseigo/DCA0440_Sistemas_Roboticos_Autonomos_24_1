import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import clear_output
from pynput import keyboard
from scipy.signal import savgol_filter

import sim


class RobotController:

####
#### Inicialização de handlers
####

    def __init__(self, clientID, robotName='Pioneer_p3dx'):
        self.clientID = clientID
        self.robotName = robotName
        self.initialize_handles()

        self.qi = self.get_robot_position()
        self.qf = []

        self.L = 0.331  # Metros
        self.r = 0.09751  # Metros
        self.prevError = np.array([0, 0])

        self.qs = np.zeros((8, 3))-5

        self.sensor_angles = np.deg2rad(np.array([-90, -50, -30, -10, 10, 30, 50, 90]))
        self.sensor_oppening_angles = np.deg2rad( np.array([20, 20, 20, 20, 20, 20, 20, 20]))
        self.orientation_history = deque(maxlen=20)

        self.map_size = np.array([-5, 5]) # Metros
        self.cell_size = 0.1
        self.grid = np.array([])
        self.map = []

        self.lx = []
        self.ly = []
        self.lt = []
        self.lxref = []
        self.lyref = []
        self.lwl = []
        self.lwr = []

        self.pathx = []
        self.pathy = []
        self.npoints = 0      
    

    def initialize_handles(self):
        self.returnCode, self.robotHandle = sim.simxGetObjectHandle(self.clientID, self.robotName, sim.simx_opmode_oneshot_wait)
        self.returnCode, self.l_wheel = sim.simxGetObjectHandle(self.clientID, self.robotName + '_leftMotor', sim.simx_opmode_oneshot_wait)
        self.returnCode, self.r_wheel = sim.simxGetObjectHandle(self.clientID, self.robotName + '_rightMotor', sim.simx_opmode_oneshot_wait)
        self.returnCode, self.goalFrame = sim.simxGetObjectHandle(self.clientID, 'Goal', sim.simx_opmode_oneshot_wait)

        self.returnCode, self.ultrasonic_sensor_1 = sim.simxGetObjectHandle(self.clientID, self.robotName + '_ultrasonicSensor1', sim.simx_opmode_blocking)
        self.returnCode, self.ultrasonic_sensor_2 = sim.simxGetObjectHandle(self.clientID, self.robotName + '_ultrasonicSensor2', sim.simx_opmode_blocking)
        self.returnCode, self.ultrasonic_sensor_3 = sim.simxGetObjectHandle(self.clientID, self.robotName + '_ultrasonicSensor3', sim.simx_opmode_blocking)
        self.returnCode, self.ultrasonic_sensor_4 = sim.simxGetObjectHandle(self.clientID, self.robotName + '_ultrasonicSensor4', sim.simx_opmode_blocking)
        self.returnCode, self.ultrasonic_sensor_5 = sim.simxGetObjectHandle(self.clientID, self.robotName + '_ultrasonicSensor5', sim.simx_opmode_blocking)
        self.returnCode, self.ultrasonic_sensor_6 = sim.simxGetObjectHandle(self.clientID, self.robotName + '_ultrasonicSensor6', sim.simx_opmode_blocking)
        self.returnCode, self.ultrasonic_sensor_7 = sim.simxGetObjectHandle(self.clientID, self.robotName + '_ultrasonicSensor7', sim.simx_opmode_blocking)
        self.returnCode, self.ultrasonic_sensor_8 = sim.simxGetObjectHandle(self.clientID, self.robotName + '_ultrasonicSensor8', sim.simx_opmode_blocking)

    def set_map_size(self, height, width):
        self.map_size = np.array([height,width])

####
#### Metodos de captura automática do mapa  

    def get_object_bounding_box(self, obj_handle):
        res, min_x = sim.simxGetObjectFloatParameter(self.clientID, obj_handle, sim.sim_objfloatparam_modelbbox_min_x, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print(f'Erro ao obter min_x para o objeto {obj_handle}')
            return None, None
        res, max_x = sim.simxGetObjectFloatParameter(self.clientID, obj_handle, sim.sim_objfloatparam_modelbbox_max_x, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print(f'Erro ao obter max_x para o objeto {obj_handle}')
            return None, None
        res, min_y = sim.simxGetObjectFloatParameter(self.clientID, obj_handle, sim.sim_objfloatparam_modelbbox_min_y, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print(f'Erro ao obter min_y para o objeto {obj_handle}')
            return None, None
        res, max_y = sim.simxGetObjectFloatParameter(self.clientID, obj_handle, sim.sim_objfloatparam_modelbbox_max_y, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print(f'Erro ao obter max_y para o objeto {obj_handle}')
            return None, None

        width = max_x - min_x
        height = max_y - min_y

        return width, height

    def is_within_area(self, x, y, x_center, y_center, distance):
        return abs(x - x_center) <= distance and abs(y - y_center) <= distance

    def get_map(self):
        # Inicializar lista de retângulos
        rectangles = []

        # Obter o handle do robô e do destino
        res, pioneer_handle = sim.simxGetObjectHandle(self.clientID, self.robotName, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print('Erro ao obter o handle do PioneerP3DX')
            pioneer_handle = None

        res, goal_handle = sim.simxGetObjectHandle(self.clientID, 'Goal', sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print('Erro ao obter o handle do Goal')
            goal_handle = None

        # Obter a posição do Pioneer e do Goal
        res, pioneer_pos = sim.simxGetObjectPosition(self.clientID, pioneer_handle, -1, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print('Erro ao obter a posição do PioneerP3DX')
            pioneer_pos = (0, 0)

        res, goal_pos = sim.simxGetObjectPosition(self.clientID, goal_handle, -1, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            print('Erro ao obter a posição do Goal')
            goal_pos = (0, 0)

        # Obter todos os objetos na simulação
        res, objects = sim.simxGetObjects(self.clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print('Número de objetos na simulação:', len(objects))

            for obj in objects:
                # Ignorar o robô e o destino
                if obj in [pioneer_handle, goal_handle]:
                    continue

                # Obter a posição do objeto
                res, position = sim.simxGetObjectPosition(self.clientID, obj, -1, sim.simx_opmode_blocking)
                if res == sim.simx_return_ok:
                    #print(f'Posição do objeto {obj}:', position)

                    # Obter as dimensões do objeto
                    width, height = self.get_object_bounding_box(obj)
                    if (width is not None and width > 0.1) and (height is not None and height > 0.1):
                        #print(f'Dimensões do objeto {obj} - Largura: {width}, Altura: {height}')
                        
                        # Filtrar objetos que podem ser quadrantes indesejados
                        if width > 2 and height > 2:  # Ajuste os critérios conforme necessário
                            print(f'Ignorando o objeto {obj} por ser um quadrante indesejado')
                            continue

                        # Verificar se o objeto está dentro da área ao redor do Pioneer e do Goal
                        if (self.is_within_area(position[0], position[1], pioneer_pos[0], pioneer_pos[1], 1) or
                            self.is_within_area(position[0], position[1], goal_pos[0], goal_pos[1], 1)):
                            print(f'Ignorando o objeto {obj} por estar dentro da área ao redor do Pioneer ou Goal')
                            continue

                        # Adicionar retângulo à lista (x, y, largura, altura)
                        rectangles.append((position[0], position[1], width, height))
                    else:
                        print(f'Erro ao obter as dimensões do objeto {obj}')
                else:
                    print(f'Erro ao obter a posição do objeto {obj}')
        else:
            print('Erro ao obter objetos:', res)

        self.map = rectangles.copy()

####
#### Métodos de controle e posicionamento
####

    def set_goal(self, qgoal):
        self.returnCode = sim.simxSetObjectPosition(self.clientID, self.goalFrame, -1, [qgoal[0], qgoal[1], 0], sim.simx_opmode_oneshot_wait)
        self.returnCode = sim.simxSetObjectOrientation(self.clientID, self.goalFrame, -1, [0, 0, qgoal[2]], sim.simx_opmode_oneshot_wait)
        self.qf = qgoal.copy()

    def get_positioning_control(self, q, ref, dt):
        dx = ref[0] - q[0]
        dy = ref[1] - q[1]

        theta_ref = np.arctan2(dy, dx)
        erro_theta = self.angle_wrap(theta_ref - q[2])

        dl = np.sqrt(dx**2 + dy**2) * np.cos(erro_theta)

        d_erro_theta = (erro_theta - self.prevError[0]) * dt
        d_dl = (dl - self.prevError[1]) * dt

        k1 = 0.8
        k2 = 0.8
        kd_theta = 2.4  # Ganho derivativo de orientação
        kd_l = 0.6   # Ganho derivativo de posição

        alpha = 1.2  # Escolher fator de filtro
        d_erro_theta_filtered = alpha * d_erro_theta + (1 - alpha) * self.prevError[0]
        d_dl_filtered = alpha * d_dl + (1 - alpha) * self.prevError[1]

        v = k1 * dl + kd_l * d_dl_filtered
        w = k2 * erro_theta + kd_theta * d_erro_theta_filtered

        self.prevError = [d_erro_theta_filtered, d_dl_filtered]

        return v, w

    @staticmethod
    def angle_wrap(angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
  
    def get_robot_position(self):
        returnCode, pos = sim.simxGetObjectPosition(self.clientID, self.robotHandle, -1, sim.simx_opmode_streaming)
        returnCode, ori = sim.simxGetObjectOrientation(self.clientID, self.robotHandle, -1, sim.simx_opmode_blocking)
        return np.array([pos[0], pos[1], ori[2]])
    
    def get_filtered_orientation(self):
        returnCode, pos = sim.simxGetObjectPosition(self.clientID, self.robotHandle, -1, sim.simx_opmode_buffer)
        returnCode, ori = sim.simxGetObjectOrientation(self.clientID, self.robotHandle, -1, sim.simx_opmode_blocking)
        
        current_orientation = ori[2]
        self.orientation_history.append(current_orientation)
        filtered_orientation = sum(self.orientation_history) / len(self.orientation_history)
        return filtered_orientation

    def control_loop(self, t_limit=100, point_change_interval=0.7):
        PathPoint = 0
        startTime = time.time()
        lastTime = startTime
        last_point_change_time = startTime
        t = 0

        while True:
            now = time.time()
            dt = now - lastTime

            q = self.get_robot_position()
            self.lx.append(q[0])
            self.ly.append(q[1])
            self.lt.append(q[2])

            v, w = self.get_positioning_control(q, [self.pathx[int(PathPoint)], self.pathy[int(PathPoint)]], dt)

            if now - last_point_change_time >= point_change_interval:
                PathPoint += 1
                last_point_change_time = now

            wr = ((2.0 * v) + (w * self.L)) / (2.0 * self.r)
            wl = ((2.0 * v) - (w * self.L)) / (2.0 * self.r)

            sim.simxSetJointTargetVelocity(self.clientID, self.r_wheel, wr, sim.simx_opmode_oneshot_wait)
            sim.simxSetJointTargetVelocity(self.clientID, self.l_wheel, wl, sim.simx_opmode_oneshot_wait)

            self.lwl.append(wl)
            self.lwr.append(wr)

            t += dt
            lastTime = now

            if np.sqrt((self.qf[0] - q[0])**2 + (self.qf[1] - q[1])**2) <= 0.001 or PathPoint > self.npoints - 1 or t > t_limit:
                sim.simxSetJointTargetVelocity(self.clientID, self.r_wheel, 0, sim.simx_opmode_oneshot_wait)
                sim.simxSetJointTargetVelocity(self.clientID, self.l_wheel, 0, sim.simx_opmode_oneshot_wait)
                break

    def manual_control_loop(self):

        # Variáveis de controle
        forward_velocity = 2.0
        turn_velocity = 1.5

        running = True
        z = np.ones(8)*1.5
        
        def on_press(key):
    
            try:
                if isinstance(key, keyboard.KeyCode):
                    if key.char == 'w':
                        sim.simxSetJointTargetVelocity(self.clientID, self.l_wheel, forward_velocity, sim.simx_opmode_streaming)
                        sim.simxSetJointTargetVelocity(self.clientID, self.r_wheel, forward_velocity, sim.simx_opmode_streaming)
                    elif key.char == 's':
                        sim.simxSetJointTargetVelocity(self.clientID, self.l_wheel, -forward_velocity, sim.simx_opmode_streaming)
                        sim.simxSetJointTargetVelocity(self.clientID, self.r_wheel, -forward_velocity, sim.simx_opmode_streaming)
                    elif key.char == 'a':
                        sim.simxSetJointTargetVelocity(self.clientID, self.l_wheel, -turn_velocity, sim.simx_opmode_streaming)
                        sim.simxSetJointTargetVelocity(self.clientID, self.r_wheel, turn_velocity, sim.simx_opmode_streaming)
                    elif key.char == 'd':
                        sim.simxSetJointTargetVelocity(self.clientID, self.l_wheel, turn_velocity, sim.simx_opmode_streaming)
                        sim.simxSetJointTargetVelocity(self.clientID, self.r_wheel, -turn_velocity, sim.simx_opmode_streaming)
            except AttributeError:
                pass


        def on_release(key):
            if isinstance(key, keyboard.KeyCode) and key.char in ['w', 'a', 's', 'd']:
                sim.simxSetJointTargetVelocity(self.clientID, self.l_wheel, 0, sim.simx_opmode_streaming)
                sim.simxSetJointTargetVelocity(self.clientID, self.r_wheel, 0, sim.simx_opmode_streaming)

            if key == keyboard.Key.esc:
                nonlocal running
                running = False
                return False
            
        def read_sensor():

            while running:
                nonlocal z
                distances = []

                for i in range(3, 7):  # Para sensores de 1 a 8
                    _, _, detected_point, _, _ = sim.simxReadProximitySensor(
                        clientID=self.clientID,
                        sensorHandle=getattr(self, f'ultrasonic_sensor_{i}'), 
                        operationMode=sim.simx_opmode_buffer
                    )
                    distance = np.sqrt(detected_point[0]**2 + detected_point[1]**2 + detected_point[2]**2)
                    distance = distance if 0.0 < distance <= 1 else 1.5
                    distances.append(distance)
                    res, pos_sens = sim.simxGetObjectPosition(self.clientID, getattr(self, f'ultrasonic_sensor_{i}'), -1, sim.simx_opmode_blocking)
                    # res, ori_sens = sim.simxGetObjectOrientation(self.clientID, self.robotHandle, -1, sim.simx_opmode_blocking)
                    ori_sens = self.get_filtered_orientation()
                    # print("sensor", i,":", ori_sens)
            
                    self.qs[i-1, :2] = pos_sens[:2]  # Atualiza as primeiras duas colunas com a posição (x, y)
                    self.qs[i-1, 2] = self.angle_wrap(ori_sens + self.sensor_angles[i-1])  # Atualiza a terceira coluna com a orientação (theta)

                z[0] = distances[0]
                z[1] = distances[1]
                z[2] = distances[2]
                z[3] = distances[3]
                z[4] = distances[4]
                z[5] = distances[5]
                z[6] = distances[6]
                z[7] = distances[7]


                # res, pos_sens = sim.simxGetObjectPosition(self.clientID, self.ultrasonic_sensor_1, -1, sim.simx_opmode_streaming) 
                # res, ori_sens =  sim.simxGetObjectOrientation(self.clientID, self.robotHandle, -1, sim.simx_opmode_oneshot_wait)
                # # self.qs[:2] = pos_sens[:2]
                # # self.qs[2] = self.angle_wrap(ori_sens[2])
                # self.qs = np.array([pos_sens[0], pos_sens[1],  self.angle_wrap(ori_sens[2])])
                 # Atualiza a posição e orientação do sensor na matriz qs
               

        def update_map_grid():
            while running:
                for i in range(3, 7):
                    clear_output(wait=True)
                    self.fill_map_grid(z[i-1], i-1)
                time.sleep(0.05)

        sim.simxReadProximitySensor(self.clientID, self.ultrasonic_sensor_1, sim.simx_opmode_streaming)
        sim.simxReadProximitySensor(self.clientID, self.ultrasonic_sensor_2, sim.simx_opmode_streaming)
        sim.simxReadProximitySensor(self.clientID, self.ultrasonic_sensor_3, sim.simx_opmode_streaming)
        sim.simxReadProximitySensor(self.clientID, self.ultrasonic_sensor_4, sim.simx_opmode_streaming)
        sim.simxReadProximitySensor(self.clientID, self.ultrasonic_sensor_5, sim.simx_opmode_streaming)
        sim.simxReadProximitySensor(self.clientID, self.ultrasonic_sensor_6, sim.simx_opmode_streaming)
        sim.simxReadProximitySensor(self.clientID, self.ultrasonic_sensor_7, sim.simx_opmode_streaming)
        sim.simxReadProximitySensor(self.clientID, self.ultrasonic_sensor_8, sim.simx_opmode_streaming)

        sensor_thread = threading.Thread(target=read_sensor)
        map_thread = threading.Thread(target=update_map_grid)
    
        sensor_thread.start()
        map_thread.start()

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()  
           
        sensor_thread.join()
        map_thread.join()
        
       

####
#### Métodos de geração de trajetória
####

    @staticmethod
    def generate_path(lbd, qi, qf):
        dx = qf[0] - qi[0]
        dy = qf[1] - qi[1]
        di = np.tan(qi[2])
        df = np.tan(qf[2])

        a0 = qi[0]
        a1 = 1  # Valores chutados
        a2 = 0.01  # Valores chutados
        a3 = dx - a2 - a1

        b0 = qi[1]
        b1 = di * a1
        b2 = 3 * dy - 3 * df * dx + df * a2 - 2 * (di - df) * a1
        b3 = 3 * df * dx - 2 * dy - df * a2 - (2 * df - di) * a1

        x = a0 + a1 * lbd + a2 * lbd**2 + a3 * lbd**3
        y = b0 + b1 * lbd + b2 * lbd**2 + b3 * lbd**3
        theta = np.arctan2(b1 + 2 * b2 * lbd + 3 * b3 * lbd**2, a1 + 2 * a2 * lbd + 3 * a3 * lbd**2)

        return [x, y, theta]

    def get_generated_path_poly(self, nPoints=100):
        points = np.linspace(0, 1, nPoints)

        llx, lly, llth = [], [], []
        print(self.qi)
        print(self.qf)
        for t in points:
            x, y, th = self.generate_path(t, self.qi, self.qf)
            llx.append(x)
            lly.append(y)
            llth.append(th)

        self.pathx = llx
        self.pathy = lly
        self.npoints = nPoints

    def get_generated_path_potential(self,printmap):
        xi = 15
        eta = 2
        Q_star = 1.0
        R_switch = 1

        x = np.linspace(-5, 5, 3000)
        y = np.linspace(-5, 5, 3000)
        X, Y = np.meshgrid(x, y)

        def attractive_potential(X, Y, goal, R_switch, xi):
            dist = np.sqrt((X - goal[0])**2 + (Y - goal[1])**2)
            U_att = np.zeros_like(dist)

            mask_conic = dist > R_switch
            U_att[mask_conic] = xi * (dist[mask_conic] - R_switch)

            mask_paraboloid = dist <= R_switch
            U_att[mask_paraboloid] = 0.5 * xi * dist[mask_paraboloid]**2

            return U_att

        def repulsive_potential(X, Y, obstacles, Q_star, eta):
            U_rep = np.zeros_like(X)
            for rect in obstacles:
                x, y, width, height = rect
                dist_x = np.maximum(0, np.abs(X - x) - ((width +1)/ 2))
                dist_y = np.maximum(0, np.abs(Y - y) - ((height +1) / 2))
                dist = np.sqrt(dist_x**2 + dist_y**2)
                mask = dist < Q_star
                U_rep[mask] += 0.5 * eta * (1.0 / (dist[mask] + 0.1) - 1.0 / Q_star)**2
            return U_rep

        U_att = attractive_potential(X, Y, self.qf, R_switch, xi)
        U_rep = repulsive_potential(X, Y, self.map, Q_star, eta)
        U_tot = U_att + U_rep

        grad_U_x, grad_U_y = np.gradient(U_tot, x, y)

        def calculate_tangential_force(grad):
            tangential_force = np.array([-grad[1], grad[0]])
            return tangential_force / np.linalg.norm(tangential_force)
        
        path = [self.qi[0:2].copy()]
        current_pos = self.qi[0:2].copy()
        alpha = 0.01
        beta = 20
        threshold = 0.2
        nPoints = 0
        
        while np.linalg.norm(current_pos - self.qf[0:2]) > threshold:
            
            ix = np.argmin(np.abs(x - current_pos[0]))
            iy = np.argmin(np.abs(y - current_pos[1]))

            grad = np.array([grad_U_y[iy, ix],grad_U_x[iy, ix]])

            repulsive_force = -grad

            if (np.linalg.norm(current_pos - self.qf[0:2]) > 0.7):
                tangential_force = calculate_tangential_force(grad)
            else:
                tangential_force = 0

            total_force = repulsive_force + beta * tangential_force

            current_pos += alpha * total_force

            current_pos[0] = np.clip(current_pos[0], -4.7, 4.7)
            current_pos[1] = np.clip(current_pos[1], -4.7, 4.7)

            path.append(current_pos.copy())

            nPoints += 1
            if nPoints > 1000:
                print("Caminho não encontrado em 1000 pontos")
                break

        path = np.array(path)

        window_length = 51 
        polyorder = 7  

        smoothed_path_x = savgol_filter(path[:, 0], window_length, polyorder)
        smoothed_path_y = savgol_filter(path[:, 1], window_length, polyorder)
        smoothed_path = np.vstack((smoothed_path_x, smoothed_path_y)).T

        self.pathx = smoothed_path_x.copy()
        self.pathy = smoothed_path_y.copy()
        self.npoints = nPoints

        if(printmap):
            fig, ax1 = plt.subplots(1, 1, figsize=(18, 6))

            c = ax1.contourf(X, Y, U_tot, levels=50, cmap='jet')
            ax1.plot(path[:, 0], path[:, 1], marker='.', color='white', label='Original Path')
            ax1.plot(smoothed_path[:, 0], smoothed_path[:, 1], marker='.', color='green', label='Smoothed Path')
            ax1.plot(self.qf[0], self.qf[1], marker='*', color='green', markersize=10, label='Goal')

            ax1.set_xlim(-5, 5)
            ax1.set_ylim(-5, 5)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title('Campo de Potencial e Caminho Planejado')
            ax1.set_aspect('equal', adjustable='box')

            ax1.legend()

            fig.colorbar(c, ax=ax1)

            plt.tight_layout()

            plt.show()

    def get_generated_path_manha(self,printmap):
        
        def attractive_potential(size, goal,obstacles):

            U_tot = np.zeros((size, size), dtype=int)

            for rect in obstacles:
                x, y, width, height = rect
                y = -y
                xui = np.clip(250 + int((x+((width+1)/2)) * 50),0,500)
                xli = np.clip(250 + int((x-((width+1)/2)) * 50),0,500)

                yui = np.clip(250 + int((y+((height+1)/2)) * 50),0,500)
                yli = np.clip(250 + int((y-((height+1)/2)) * 50),0,500)

                
                U_tot[yli:yui, xli:xui] = 250000

            gxi, gyi = goal[0:2]

            rows, cols = 500,500
            stack = [(gxi,gyi)]

            wave = 0

            while stack:
                wave += 1
                current = stack.pop(0)

                r, c = current
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                
                for neighbor in neighbors:
                    nr, nc = neighbor
                    if (0 <= nr < rows and 0 <= nc < cols) and not U_tot[nr,nc]:
                        stack.append(neighbor)
                        U_tot[nr,nc] = wave

            return U_tot

        rxi = np.clip(250 + int(self.qi[0] * 50),0,500)
        ryi = np.clip(250 + int(-self.qi[1] * 50),0,500)

        gxi = np.clip(250 + int(self.qf[0] * 50),0,500)
        gyi = np.clip(250 + int(-self.qf[1] * 50),0,500)

        start = (ryi,rxi)
        end = (gyi,gxi)

        U_tot = attractive_potential(500,end,self.map)

        def find_lowest_value_path(grid, start, end, search_range):
            path = [start]
            current_point = start
            nPoints = 0

            while current_point != end:
                x, y = current_point
                
                x_min = max(0, x - search_range)
                x_max = min(grid.shape[0], x + search_range + 1)
                y_min = max(0, y - search_range)
                y_max = min(grid.shape[1], y + search_range + 1)
                
                subgrid = grid[x_min:x_max, y_min:y_max]
                
                min_index = np.unravel_index(np.argmin(subgrid, axis=None), subgrid.shape)
                min_point = (x_min + min_index[0], y_min + min_index[1])
                
                current_point = min_point
                path.append(current_point)
                
                if (x_min <= end[0] < x_max) and (y_min <= end[1] < y_max):
                    path.append(end)
                    break
                nPoints += 1
                if(nPoints>=1000):
                    break
            
            return path, nPoints
        
        
        path,self.npoints = find_lowest_value_path(U_tot, start,end,5)

        path_x, path_y = zip(*path) if path else ([], [])

        path_x = np.array(path_x)
        path_y = np.array(path_y)

        self.pathx = path_y.copy()
        self.pathx = np.clip((self.pathx - 250)/50,-5,5)

        self.pathy = path_x.copy()
        self.pathy = - np.clip((self.pathy - 250)/50,-5,5)

        if(printmap):
            # Separate path coordinates
            

            # Plot the grid
            plt.figure(figsize=(8, 8))
            plt.imshow(U_tot, cmap='viridis', origin='upper', interpolation='none')
            plt.colorbar(label='Poder Repulsivo')
            plt.title('Mapa Manhatan')

            # Plot the path
            plt.plot(path_y, path_x, color='red', marker='o', markersize=3, linewidth=2, linestyle='-', alpha=0.7)

            plt.show()

    def get_generated_path_graph(self, n_grade=50, printmap=True):

        def real2grid(real_x, real_y, n):
            
            min_val, max_val = -5.0, 5.0
            
            i = np.clip(int(n/2) + int(-real_y * (n/(max_val - min_val))),0,n-1)
            j = np.clip(int(n/2) + int(real_x * (n/(max_val - min_val))),0,n-1)

            return (i, j)

        def grid2real(i, j, n):
            
            min_val, max_val = -5.0, 5.0

            real_x = np.clip((j - n/2)/(n/(max_val - min_val)),min_val,max_val)
            real_y = -np.clip((i - n/2)/(n/(max_val - min_val)),min_val,max_val)

            return real_x, real_y
        
        n = n_grade

        grid = np.zeros((n, n))

        for rect in self.map:
            x, y, width, height = rect
            y = -y
            xui = int(np.clip(n/2 + int((x+((width+1)/2)) * n/10),0,n-1))
            xli = int(np.clip(n/2 + int((x-((width+1)/2)) * n/10),0,n-1))

            yui = int(np.clip(n/2 + int((y+((height+1)/2)) * n/10),0,n-1))
            yli = int(np.clip(n/2 + int((y-((height+1)/2)) * n/10),0,n-1))

            
            grid[yli:yui, xli:xui] = 1

        G = nx.grid_2d_graph(n, n) 
        node_colors = {node: 'green' for node in G.nodes()}

        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1 or r == 0 or r == n-1 or c == 0 or c == n-1:  
                    G.remove_node((r,c))
        
         
        start_node = real2grid(self.qi[0], self.qi[1], n)
        node_colors[start_node] = 'red'
        end_node = real2grid(self.qf[0], self.qf[1], n)
        node_colors[end_node] = 'yellow'

        path = nx.shortest_path(G, source=start_node, target=end_node)

        for node in path:
            node_colors[node] = 'blue'

        node_colors_list = [node_colors.get(node, 'green') for node in G.nodes()]

        if(printmap):
            node_colors_list = [node_colors[node] for node in G.nodes()]

            fig = plt.figure(figsize=(10,10), dpi=100)
            ax = fig.add_subplot(111, aspect='equal')

            ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
            obj = ax.imshow(grid, cmap='Greys')

            posn = {node: (node[1],node[0]) for node in G.nodes()}
            nx.draw(G, posn, font_size=0, node_color=node_colors_list, with_labels=False, node_size=10, ax=ax)            
            plt.draw()


        x_path = []
        y_path = []

        for pos in path:
            px, py = grid2real(pos[0], pos[1], n)
            x_path.append(px)
            y_path.append(py)
        
        self.pathx = x_path.copy()
        self.pathy = y_path.copy()
        self.npoints = len(x_path)

    def send_generated_path(self):

        returnCode, pathHandle = sim.simxGetObjectHandle(self.clientID, "FloorPath", sim.simx_opmode_oneshot_wait)
        res = sim.simxSetObjectPosition(self.clientID, pathHandle, -1, [self.qi[0], self.qi[1], 0], sim.simx_opmode_oneshot)
        

        for x, y in zip(self.pathx, self.pathy):
            res = sim.simxSetObjectPosition(self.clientID, pathHandle, -1, [x, y, 0], sim.simx_opmode_oneshot)
            time.sleep(0.07)
        return res
    
####
#### Geração do mapa
####

    def clear_map_grid(self):

        # res, pos_sens = sim.simxGetObjectPosition(self.clientID, self.ultrasonic_sensor_1, -1, sim.simx_opmode_streaming) 
        # res, ori_sens =  sim.simxGetObjectOrientation(self.clientID, self.robotHandle, -1, sim.simx_opmode_oneshot_wait)
        # # self.qs[2] = self.angle_wrap(ori_sens[2])
        # self.qs = np.array([pos_sens[0], pos_sens[1], self.angle_wrap(ori_sens[2])])
        for i in range(3, 7):  # Para sensores de 1 a 8
            res, pos_sens = sim.simxGetObjectPosition(self.clientID, getattr(self, f'ultrasonic_sensor_{i}'), -1, sim.simx_opmode_blocking)
            res, ori_sens = sim.simxGetObjectOrientation(self.clientID, self.robotHandle, -1, sim.simx_opmode_blocking)
            self.qs[i-1, :2] = pos_sens[:2]  # Atualiza as primeiras duas colunas com a posição (x, y)
            self.qs[i-1, 2] = self.angle_wrap(ori_sens[2] + self.sensor_angles[i-1])  # Atualiza a terceira coluna com a orientação (theta)
     
        cx = np.arange(self.map_size[0], self.map_size[1]+self.cell_size, self.cell_size)
        cy = np.arange(self.map_size[1], self.map_size[0]-self.cell_size, -self.cell_size)
      
        col = len(cx)
        lin = len(cy)
    
        self.grid = np.ones((col, lin)) * 50

        plt.imshow(self.grid, cmap='Greys', vmin=0, vmax=100, extent=[-5, 5, -5, 5])
        plt.show()
    
    def fill_map_grid(self, z, i):

        def check_angle(ang_l, ang_h, phi):
            if ang_l < ang_h:
                if ang_l <= phi <= ang_h:
                    return True
            elif ang_l > ang_h:
                if ang_l <= phi or phi <= ang_h:
                    return True
            return False
        
        def cell_value_sat(value):
            if value < 0.0:
                return 0.0
            elif value > 100.0:
                return 100.0
            return value
        
        z_max = 1.5 # m
        alpha = self.sensor_oppening_angles[i]
        e = 0.15 + 0.05 * np.random.randn()

        r_l = (z - e/2)
        r_h = (z + e/2)

        # ang_l = (self.angle_wrap(self.qs[i, 2]) - alpha/2)
        # ang_h = (self.angle_wrap(self.qs[i, 2]) + alpha/2)

        # ang_l = self.angle_wrap(ang_l)
        # ang_h = self.angle_wrap(ang_h)

        cx = np.arange(self.map_size[0], self.map_size[1]+self.cell_size, self.cell_size)
        cy = np.arange(self.map_size[1], self.map_size[0]-self.cell_size, -self.cell_size)

        sim.simxGetObjectPosition(self.clientID, self.robotHandle, -1, sim.simx_opmode_streaming)
        returnCode, pos = sim.simxGetObjectPosition(self.clientID, self.robotHandle, -1, sim.simx_opmode_buffer)
        ori = self.get_filtered_orientation()
        
        xr, yr, tr = pos[0], pos[1], ori
        for yi in range(self.grid.shape[1]):
            for xi in range(self.grid.shape[0]):

                xm, ym = cx[xi], cy[yi]
                
                # time.sleep(0.1)
                # print(xr,yr,tr)

                dist = np.sqrt((xm-xr)**2 + (ym-yr)**2)
                phi = self.angle_wrap(np.arctan2((ym-yr)**2, (xm-xr)**2) - (self.qs[i, 2]))
                print(phi)

                # dist = np.sqrt((xm-self.qs[i, 0])**2 + (ym-self.qs[i, 1])**2)
                # phi = np.arctan2((ym-self.qs[i, 1]), (xm-self.qs[i, 0]))
                # print(z)

                # if z < 0.001:
                #     continue
                # if z > 2:
                #     if 0.01 <= dist <= 1:
                #         if check_angle(ang_h, ang_l, phi):
                #             self.grid[yi][xi] = cell_value_sat(self.grid[yi][xi] - 50)
                #             continue

                if (dist > min(z_max, r_h)) or (abs(phi) > alpha/2):
                    # print("Aqui1")
                    # print("C1:", dist > min(z_max, r_h))
                    # print(dist, min(z_max, r_h) )
                    continue

                elif (z < z_max) and (abs(dist - z) < e/2) and (abs(phi) < alpha/2):
                    self.grid[yi][xi] = cell_value_sat(self.grid[yi][xi] + 30)
                    # continue
                    # print("Aqui2")

                elif (dist <= r_l) and (abs(phi) < alpha/2):
                    self.grid[yi][xi] = cell_value_sat(self.grid[yi][xi] - 15)
                    # continue
                    # print("Aqui3")

                

                # if dist < 0.01:
                #     self.grid[yi][xi] = cell_value_sat(self.grid[yi][xi] - 15)
                # elif dist > 1:
                #     continue
                #     # self.grid[yi][xi] = cell_value_sat(self.grid[yi][xi] + 0)
                # elif r_l <= dist <= r_h:
                #     if check_angle(ang_l, ang_h, phi):
                #         self.grid[yi][xi] = cell_value_sat(self.grid[yi][xi] + 30)
                # elif 0.01 < dist < r_l:
                #     if check_angle(ang_l, ang_h, phi):
                #         self.grid[yi][xi] = cell_value_sat(self.grid[yi][xi] - 15)
                # elif r_h < dist < 1.0:
                #     continue
                #     # self.grid[yi][xi] = cell_value_sat(self.grid[yi][xi] + 0 )
            
            
        plt.imshow(self.grid, cmap='Greys', vmin=0, vmax=100, extent=[-5, 5, -5, 5])
        plt.show()

####
#### PLOTS dos dados
####


    def plot_vel_rodas(self):
        plt.title("Velocidades das rodas")
        plt.plot(self.lwl, 'r-', label='$w_l$')
        plt.plot(self.lwr, 'b-', label='$w_r$')
        plt.xlabel("t")
        plt.ylabel("rad/s")
        plt.legend()
        plt.show()

    def plot_pose(self):
        plt.title("Posição e orientação do robô ao longo do tempo")
        plt.plot(self.lx, 'r-', label='$x(t)$ [m]')
        plt.plot(self.ly, 'b-', label='$y(t)$ [m]')
        plt.plot(self.lt, 'g-', label='$\\Theta(t)$ [rad]')
        plt.xlabel("t")
        plt.legend()
        plt.show()

    def plot_caminho(self):
        plt.title("Caminho seguido pelo robô no plano $xy$")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(self.lx, self.ly, "--y", label="Caminho Realizado")
        plt.plot(self.pathx, self.pathy, "k--", label="Caminho Gerado")
        plt.plot(self.lx[0], self.ly[0], "*b", label="Inicio")
        plt.plot(self.qf[0], self.qf[1], "xr", label="Goal")
        plt.plot(self.lx[-1], self.ly[-1], "*g", label="Fim")
        plt.axis('equal')
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_map(self):
        # Criar os subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Espaço de ocupação no primeiro subplot
        for rect in self.map:
            x, y, width, height = rect
            ax1.add_patch(plt.Rectangle((x - width/2, y - height/2), width, height, edgecolor='r', facecolor='r'))  # Retângulos sólidos

        # Marcar a posição do Pioneer com um quadrado azul
        pioneer_square = plt.Rectangle((self.qi[0] - 0.5, self.qi[1] - 0.5), 1, 1, edgecolor='blue', facecolor='blue')
        ax1.add_patch(pioneer_square)
        print(self.qf)
        # Marcar a posição do Goal com uma estrela verde
        ax1.plot(self.qf[0], self.qf[1], marker='*', color='green', markersize=10, label='Goal')

        ax1.set_xlim(-5, 5)  # Ajuste os limites conforme necessário
        ax1.set_ylim(-5, 5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Espaço de Ocupação no Plano XY')
        ax1.set_aspect('equal', adjustable='box')

        # Adicionar legenda ao primeiro subplot
        ax1.legend()
        # Espaço de configuração no segundo subplot
        for rect in self.map:
            x, y, width, height = rect
            width = width + 1
            height = height + 1
            ax2.add_patch(plt.Rectangle((x - width/2, y - height/2), width, height, edgecolor='r', facecolor='r'))  # Retângulos sólidos

        # Marcar a posição do Pioneer com um triângulo azul
        ax2.plot(self.qi[0], self.qi[1], marker='.', color='blue', markersize=10, label='Pioneer')
        
        # Marcar a posição do Goal com uma estrela verde
        ax2.plot(self.qf[0], self.qf[1], marker='*', color='green', markersize=10, label='Goal')

        ax2.set_xlim(-5, 5)  # Ajuste os limites conforme necessário
        ax2.set_ylim(-5, 5)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Espaço de Configuração 2D')
        ax2.set_aspect('equal', adjustable='box')


        # Adicionar legenda ao segundo subplot
        ax2.legend()
       
        # Ajustar o layout para melhor visualização
        plt.tight_layout()

        plt.show()