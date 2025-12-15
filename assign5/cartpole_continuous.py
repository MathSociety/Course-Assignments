import numpy as np
import gymnasium as gym
from typing import Optional

class CartPoleEnv_Continuous(gym.Env):
    metadata = {
        "render_modes": ['human', 'rgb_array'],
        "render_fps": 50,
    }
    def __init__(self, control_mode, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.max_force = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 24 * 2 * np.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                1.5
            ],
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(-self.max_force, self.max_force, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        
        self.state = None
        self.pos_desired = None
        
        if control_mode!='setpoint' and control_mode!='regulatory':
            print('Error in control mode. By default we assume regulatory.')
            self.control_mode = 'regulatory'
        else:
            self.control_mode = control_mode
        
        self.counter = 0
        self.steps_beyond_terminated = None
        
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

    def step(self, action):
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot, _ = self.state        
        if np.isscalar(action):
            force = max(-self.max_force,min(self.max_force, action))
        else:
            force = max(-self.max_force,min(self.max_force, action[0]))
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        
        self.state = (x, x_dot, theta, theta_dot, self.pos_desired)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            if x>=self.pos_desired-0.25 and x<=self.pos_desired+0.25:
                reward = 1.0
            else:
                reward = 0.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0
        
        if self.render_mode == "human":
            self.render()
        
        self.counter+=1
        if self.counter>=500:
            truncated = True
        else:
            truncated = False
        
        observation = self.get_observation()
        
        return observation, reward, terminated, truncated, {}

    # def reset(self):
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = np.zeros(5)
        self.state[:4] = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        
        if self.control_mode=='setpoint':
            self.pos_desired = np.random.uniform(low=-1.5, high=1.5)
        else:
            self.pos_desired = 0
            
        self.state[4] = self.pos_desired
        self.counter = 0
        
        self.steps_beyond_terminated = None
        
        if self.render_mode == "human":
            self.render()            
        
        observation = self.get_observation()
        
        return observation, {}
    
    
    def get_observation(self):
        x, x_dot, theta, theta_dot, pos_desired = self.state
        
        x_obs = x + np.random.normal(loc=0, scale=0.5)
        x_dot_obs = x_dot + np.random.normal(loc=0, scale=0.5)
        theta_obs = theta + np.random.normal(loc=0, scale=0.1)
        theta_dot_obs = theta_dot + np.random.normal(loc=0, scale=0.1)
        
        return np.array([x_obs, x_dot_obs, theta_obs, theta_dot_obs, pos_desired])
    
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))
        
        
        region = int(scale*(self.pos_desired-0.25) + self.screen_width / 2.0)
        gfxdraw.vline(self.surf, region, 0, self.screen_height, (0, 200, 0))
        
        region = int(scale*(self.pos_desired+0.25) + self.screen_width / 2.0)
        gfxdraw.vline(self.surf, region, 0, self.screen_height, (0, 200, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False