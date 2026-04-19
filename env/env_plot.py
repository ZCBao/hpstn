import imageio
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
from math import cos, sin

class EnvPlot:
    def __init__(self, components=dict()):
        self.fig, self.ax = plt.subplots()
        
        self.map_matrix = components['map_matrix']
        if self.map_matrix is not None:
            self.map_origin = components['map_origin']
            self.map_width = components['map_matrix'].shape[0] * components['map_resolution']
            self.map_height = components['map_matrix'].shape[1] * components['map_resolution']
        self.world_width = components['world_width']
        self.world_height = components['world_height']
        self.world_origin = components['world_origin']
        self.components = components

        self.dynamic_obstacle_plot_list = []
        self.start_goal_plot_list = []
        self.robot_plot_list = []
        self.lidar_plot_list = []
        self.trajectory_plot_list = []
        if components['robots'].robot_num <= 10:
            self.color_list = plt.get_cmap('tab10')
        else:
            self.color_list = plt.get_cmap('tab20')

        self.init_plot()

    # draw ax
    def init_plot(self):
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.world_origin[0], self.world_origin[0] + self.world_width)
        self.ax.set_ylim(self.world_origin[1], self.world_origin[1] + self.world_height)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

        self.draw_static_components()
            
    def draw_static_components(self):
        if self.map_matrix is not None:
            self.ax.imshow(self.map_matrix.T, cmap='Greys', origin='lower', 
                           extent=[self.map_origin[0], self.map_origin[0]+self.map_width, self.map_origin[1], self.map_origin[1]+self.map_height])     
        self.draw_static_obstacles_circle(self.components['obstacles'])
        self.draw_obstacles_line(self.components['obstacles'])
        self.draw_obstacles_polygon(self.components['obstacles'])

    def draw_dynamic_components(self, save_step=None, step_time=None, **kwargs):
        self.ax.set_title('Time: {:.2f} s'.format(save_step*step_time))
        self.draw_dynamic_obstacles_circle(self.components['obstacles'])
        self.draw_robots(self.components['robots'], **kwargs)

    def draw_static_obstacles_circle(self, obstacles):
        for obstacle_circle in obstacles.obstacle_circle_static_list:
            self.draw_static_obstacle_circle(obstacle_circle)

    def draw_obstacles_line(self, obstacles):
        for obstacle_line in obstacles.obstacle_line_list:
            self.draw_obstacle_line(obstacle_line)

    def draw_obstacles_polygon(self, obstacles):
        for obstacle_polygon in obstacles.obstacle_polygon_list:
            self.draw_obstacle_polygon(obstacle_polygon)

    def draw_dynamic_obstacles_circle(self, obstacles):
        for obstacle_circle in obstacles.obstacle_circle_dynamic_list:
            self.draw_dynamic_obstacle_circle(obstacle_circle)

    def draw_robots(self, robots, **kwargs):
        for robot in robots.robot_list:
            self.draw_robot(robot, **kwargs)

    def draw_static_obstacle_circle(self, obstacle_circle, obstacle_circle_color='k'):
        if obstacle_circle.type == 'static':
            x = obstacle_circle.state[0]
            y = obstacle_circle.state[1]
            
            circle = matplotlib.patches.Circle(xy=(x, y), radius=obstacle_circle.radius, color=obstacle_circle_color)
            self.ax.add_patch(circle)
    
    def draw_obstacle_line(self, obstacle_line, obstacle_line_color='k'):
        x_list = [obstacle_line.vertex_list[0][0], obstacle_line.vertex_list[1][0]]
        y_list = [obstacle_line.vertex_list[0][1], obstacle_line.vertex_list[1][1]]
        self.ax.plot(x_list, y_list, color=obstacle_line_color)

    def draw_obstacle_polygon(self, obstacle_polygon, obstacle_polygon_color='k'):
        polygon = matplotlib.patches.Polygon(obstacle_polygon.vertex_list, color=obstacle_polygon_color)
        self.ax.add_patch(polygon)

    def draw_dynamic_obstacle_circle(self, obstacle_circle, obstacle_circle_color='b'):
        if obstacle_circle.type == 'dynamic':
            x = obstacle_circle.state[0]
            y = obstacle_circle.state[1]
            
            circle = matplotlib.patches.Circle(xy=(x, y), radius=obstacle_circle.radius, color=obstacle_circle_color)
            circle.set_zorder(2)
            self.ax.add_patch(circle)

            self.dynamic_obstacle_plot_list.append(circle)

    def draw_robot(self, robot, robot_color='g', show_lidar=False, show_trajectory=False):
        start_x = robot.start[0]
        start_y = robot.start[1]
        goal_x = robot.goal[0]
        goal_y = robot.goal[1]
        x = robot.state[0]
        y = robot.state[1]

        # start_point = self.ax.plot(start_x, start_y, 'r.', markersize=10, alpha=0.5)
        # self.ax.text(start_x + 0.3, start_y + 0.2, 's' + str(robot.id), fontsize=10, color='k')
        # self.start_goal_plot_list.append(start_point)

        goal_point = self.ax.plot(goal_x, goal_y, 'r*', markersize=12)
        self.ax.text(goal_x, goal_y - 0.5, 'g' + str(robot.id), fontsize=10, color='k')
        self.start_goal_plot_list.append(goal_point)

        if robot.shape == 'circle':
            robot_circle = matplotlib.patches.Circle(xy=(x, y), radius=robot.radius, color=robot_color)
            robot_circle.set_zorder(4)
            self.ax.add_patch(robot_circle)
            self.robot_plot_list.append(robot_circle)
        elif robot.shape == 'polygon':
            robot_polygon = matplotlib.patches.Polygon(robot.vertex_list, color=robot_color)
            robot_polygon.set_zorder(4)
            self.ax.add_patch(robot_polygon)
            self.robot_plot_list.append(robot_polygon)
            robot_circle = matplotlib.patches.Circle(xy=(x, y), radius=robot.radius, color=robot_color, fill=False)
            robot_circle.set_zorder(4)
            self.ax.add_patch(robot_circle)
            self.robot_plot_list.append(robot_circle)
        self.ax.text(x - 0.2, y + robot.radius + 0.2, 'r' + str(robot.id), fontsize=10, color='k')

        yaw = robot.state[2]
        arrow = matplotlib.patches.Arrow(x, y, (robot.radius+0.3)*cos(yaw), (robot.radius+0.3)*sin(yaw), width=0.5)
        arrow.set_zorder(3)
        self.ax.add_patch(arrow)
        self.robot_plot_list.append(arrow)

        if robot.lidar is not None and (show_lidar == -1 or show_lidar == robot.id):
            for end_point in robot.lidar.end_points[:, :]:
                x_value = [robot.lidar.start_point[0], end_point[0]]
                y_value = [robot.lidar.start_point[1], end_point[1]]
                lidar_line = self.ax.plot(x_value, y_value, color='b', alpha=0.5)
                self.lidar_plot_list.append(lidar_line)

        if show_trajectory:
            x_list = [robot.previous_state[0], robot.state[0]]
            y_list = [robot.previous_state[1], robot.state[1]]
            trajectory_line = self.ax.plot(x_list, y_list, '-', color=self.color_list(robot.id), linewidth=1)
            self.trajectory_plot_list.append(trajectory_line)
    
    def clear_dynamic_components(self, new_episode=False):
        self.ax.texts.clear()

        for dynamic_obstacle_plot in self.dynamic_obstacle_plot_list:
            dynamic_obstacle_plot.remove()

        for start_goal_plot in self.start_goal_plot_list:
            start_goal_plot.pop(0).remove()

        for robot_plot in self.robot_plot_list:
            robot_plot.remove()

        for lidar_plot in self.lidar_plot_list:
            lidar_plot.pop(0).remove()

        self.dynamic_obstacle_plot_list = []
        self.start_goal_plot_list = []
        self.robot_plot_list = []
        self.lidar_plot_list = []

        if new_episode:
            for trajectory_plot in self.trajectory_plot_list:
                trajectory_plot.pop(0).remove()
            self.trajectory_plot_list = []

    def save_figure(self, save_path, save_episode, save_step, format='png'):
        save_path = os.path.join(save_path, 'episode_' + str(save_episode))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        elif save_step == 0:
            images = [img for img in os.listdir(save_path) if img.endswith(".png")]
            for img in images:
                os.remove(os.path.join(save_path, img))
        save_id = str(save_step).zfill(3)
        save_name = os.path.join(save_path, save_id + '.' + format)
        plt.savefig(save_name, format=format, bbox_inches='tight')

    def save_gif(self, save_path, save_episode, step_time=0.1, clear_figure=True):
        save_path = os.path.join(save_path, 'episode_' + str(save_episode))
        assert os.path.exists(save_path)
        images = [img for img in os.listdir(save_path) if img.endswith(".png")]
        images.sort()
        image_list = []
        for img in images:
            image_list.append(imageio.imread(os.path.join(save_path, img)))
        
        save_name = os.path.join(save_path, 'episode_' + str(save_episode) + '.gif')
        imageio.mimsave(save_name, image_list, duration=step_time*1000)

        if clear_figure:
            for img in images:
                os.remove(os.path.join(save_path, img))
    
    def save_video(self, save_path, save_episode, step_time=0.1, clear_figure=True):
        save_path = os.path.join(save_path, 'episode_' + str(save_episode))
        assert os.path.exists(save_path)
        images = [img for img in os.listdir(save_path) if img.endswith(".png")]
        images.sort()
        image_list = []
        for img in images:
            image_list.append(imageio.imread(os.path.join(save_path, img)))

        save_name = os.path.join(save_path, 'episode_' + str(save_episode) + '.mp4')
        imageio.mimsave(save_name, image_list, fps=1/step_time)

        if clear_figure:
            for img in images:
                os.remove(os.path.join(save_path, img))
    
    def pause(self, interval=0.001):
        plt.pause(interval)