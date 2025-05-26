import numpy as np
import matplotlib.pyplot as plt

class MobiusStrip:
    def __init__(self, R, w, n):
        self.R = R  # Radius from center to strip
        self.w = w  # Width of the strip
        self.n = n  # Resolution (number of points)
        self.u_values = np.linspace(0, 2 * np.pi, n)
        self.v_values = np.linspace(-w / 2, w / 2, n)
        self.x, self.y, self.z = self.create_mesh()

    def create_mesh(self):
        u, v = np.meshgrid(self.u_values, self.v_values)
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def compute_surface_area(self):
        du = self.u_values[1] - self.u_values[0]
        dv = self.v_values[1] - self.v_values[0]

        dx_du = np.gradient(self.x, du, axis=1)
        dx_dv = np.gradient(self.x, dv, axis=0)
        dy_du = np.gradient(self.y, du, axis=1)
        dy_dv = np.gradient(self.y, dv, axis=0)
        dz_du = np.gradient(self.z, du, axis=1)
        dz_dv = np.gradient(self.z, dv, axis=0)

        normal_x = dy_du * dz_dv - dz_du * dy_dv
        normal_y = dz_du * dx_dv - dx_du * dz_dv
        normal_z = dx_du * dy_dv - dy_du * dx_dv

        area_element = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        surface_area = np.sum(area_element) * du * dv
        return surface_area

    def compute_edge_length(self):
        u = self.u_values
        v = self.w / 2

        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)

        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        distances = np.sqrt(dx**2 + dy**2 + dz**2)

        total_edge_length = np.sum(distances) * 2
        return total_edge_length

    def plot_strip(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, color='lightblue', edgecolor='black', alpha=0.8)
        ax.set_title("Mobius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    strip = MobiusStrip(R=1, w=0.3, n=100)
    print("Surface Area:", round(strip.compute_surface_area(), 4))
    print("Edge Length:", round(strip.compute_edge_length(), 4))
    strip.plot_strip()
