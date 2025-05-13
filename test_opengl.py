import pyglet
from pyglet.gl import *

# Create a window
window = pyglet.window.Window(width=800, height=600, caption='OpenGL Test')

# Enable alpha blending
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Set clear color to dark blue
glClearColor(0.1, 0.1, 0.2, 1.0)

# Triangle vertices (x, y, z, r, g, b, a)
vertices = [
    # Position    # Color
    0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0,  # Top vertex (red)
    -0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0,  # Bottom left (green)
    0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 1.0,  # Bottom right (blue)
]

# Create vertex list
vertex_list = pyglet.graphics.vertex_list(3, ('v3f', vertices[::7]), ('c4f', vertices[3:]))

# Rotation angle
angle = 0

@window.event
def on_draw():
    window.clear()
    
    # Save the current matrix
    glPushMatrix()
    
    # Rotate around the z-axis
    glRotatef(angle, 0, 0, 1)
    
    # Draw the triangle
    vertex_list.draw(GL_TRIANGLES)
    
    # Restore the matrix
    glPopMatrix()

def update(dt):
    global angle
    angle += 30 * dt  # Rotate 30 degrees per second

# Schedule the update function
pyglet.clock.schedule_interval(update, 1/60.0)

# Run the application
pyglet.app.run() 