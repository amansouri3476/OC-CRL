import numpy as np
import pygame
from pygame import gfxdraw

COLOURS_ = [
    [2, 156, 154],
    [222, 100, 100],
    [149, 59, 123],
    [74, 114, 179],
    [27, 159, 119],
    [218, 95, 2],
    [117, 112, 180],
    [232, 41, 139],
    [102, 167, 30],
    [231, 172, 2],
    [167, 118, 29],
    [102, 102, 102],
]

SHAPES_ = [
    "circle",
    "square",
    "triangle",
    "heart"
]

def draw_shape(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=64,
    y_shift=0.0,
    offset=None,
    shape="circle"
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset
    if shape == "circle":
        gfxdraw.aacircle(
            surf, int(x), int(y - offset * y_shift), int(radius * scale), color
            )
        gfxdraw.filled_circle(
            surf, int(x), int(y - offset * y_shift), int(radius * scale), color
        )
    elif shape == "square":
        radius = int(radius * scale)*2
        pygame.draw.rect(surface=surf, color=color,
                        rect=(int(x) - radius//2, int(y - offset * y_shift) - radius//2, radius, radius))
    elif shape == "triangle":
        radius = (radius * scale)*2
        x, y = ((x) - radius/2, (y - offset * y_shift) - radius/2)
        pygame.draw.polygon(surface=surf, color=color,
                        points=[(int(i), int(j)) for i, j in [(x+radius//2,y+radius), (x+radius,y), (x,y)]])
    elif shape == "heart":
        radius = (radius * scale)*2
        x, y = ((x) , (y - offset * y_shift))
        s = 3.5
        j = 1.33
        pygame.draw.circle(surface=surf, color=color,
                       center=(int(x+ radius /(s * j)), int(y + radius/(s * j))), radius=int(radius/s))
        pygame.draw.circle(surface=surf, color=color,
                       center=(int(x- radius/(s*j)), int(y + radius /(s*j))), radius=int(radius/s))
        pygame.draw.polygon(surface=surf, color=color,
                        points=[(int(i), int(j)) for i, j in [(x,y-radius/2), (x-radius/2.0,y + radius/30), (x+radius/2.0,y+ radius/30)]])


def draw_scene(z, colours=None, screen_dim=64):
    pygame.init()
    screen = pygame.display.set_mode((screen_dim, screen_dim))
    surf = pygame.Surface((screen_dim, screen_dim))
    surf.fill((255, 255, 255))
    ball_rad = 0.2
    
    obj_masks = []
    if z.ndim == 1:
        z = z.reshape((1, 2))
    if colours is None:
        colours = [COLOURS_[3]] * z.shape[0]
    for i in range(z.shape[0]):
        obj_masks.append(
            draw_shape(
                z[i, 0],
                z[i, 1],
                surf,
                color=colours[i],
                radius=ball_rad,
                screen_width=screen_dim,
                y_shift=0.0,
                offset=0.0,
                shape=SHAPES_[int(z[i,3])]
            )
        )

    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    image = np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
            )
    del surf
    del screen
    pygame.quit()

    return image
