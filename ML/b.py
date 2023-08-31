from PIL import Image, ImageDraw
import numpy as np
import math 
from math import sin, radians , cos
def draw_ashoka_chakra(image_size, chakra_radius):
    image = Image.new('RGBA', image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    center_x, center_y = image_size[0] // 2, image_size[1] // 2

    # Draw the 24 spokes in the Chakra
    for i in range(24):
        angle = i * 15
        start_x = center_x + 0.45 * chakra_radius * cos(radians(angle))
        start_y = center_y + 0.45 * chakra_radius * sin(radians(angle))
        end_x = center_x + chakra_radius * cos(radians(angle))
        end_y = center_y + chakra_radius * sin(radians(angle))
        draw.line([(start_x, start_y), (end_x, end_y)], fill='blue', width=1)

    # Draw the central blue circle
    chakra_diameter = chakra_radius * 2
    chakra_bbox = (center_x - chakra_radius, center_y - chakra_radius,
                   center_x + chakra_radius, center_y + chakra_radius)
    draw.ellipse(chakra_bbox, fill='blue')

    # Draw the small blue circles in the Chakra
    small_circle_radius = chakra_radius / 7
    for i in range(4):
        for j in range(6):
            angle = (j * 15) + (i % 2) * 7.5
            x = center_x + 0.7 * chakra_radius * cos(radians(angle))
            y = center_y + 0.7 * chakra_radius * sin(radians(angle))
            small_circle_bbox = (x - small_circle_radius, y - small_circle_radius,
                                 x + small_circle_radius, y + small_circle_radius)
            draw.ellipse(small_circle_bbox, fill='blue')

    return image

if __name__ == "__main__":
    # Image size and Chakra radius
    image_size = (200, 200)  # Adjust as per your requirements
    chakra_radius = 80      # Adjust as per your requirements

    chakra_image = draw_ashoka_chakra(image_size, chakra_radius)
    chakra_image.save("ashoka_chakra.png")
    chakra_image.show()
