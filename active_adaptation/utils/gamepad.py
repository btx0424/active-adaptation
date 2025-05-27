import pygame

class Gamepad:
    
    def __init__(self):
        pygame.init()

        # Check for connected joysticks (game controllers)
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No game controller found!")

        # Initialize the first joystick
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print(f"Controller detected: {self.joystick.get_name()}")
        self.num_axes = self.joystick.get_numaxes()
        self.num_hats = self.joystick.get_numhats()
        self.num_buttons = self.joystick.get_numbuttons()
        self.update()
    
    @property
    def lxy(self):
        return self.axes[0:2]
    
    @property
    def rxy(self):
        return self.axes[2:4]
    
    @property
    def lt(self):
        return - (self.axes[4] - 1) / 2. 
    
    @property
    def rt(self):
        return - (self.axes[5] - 1) / 2.

    def update(self):
        pygame.event.pump()
        self.axes = [self.joystick.get_axis(i) for i in range(self.num_axes)]
        self.buttons = [self.joystick.get_button(i) for i in range(self.num_buttons)]
        self.hats = [self.joystick.get_hat(i) for i in range(self.num_hats)]

