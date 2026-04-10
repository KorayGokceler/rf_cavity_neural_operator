import numpy as np
import matplotlib.pyplot as plt
import os

def generate_geometry(method, cx=0.05, cy=0.05):
    if method == 'sharp':
        n_pts = np.random.randint(7, 13)
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_pts))
        r = np.random.uniform(0.02, 0.046, n_pts)
        pts_c = [(cx + ri*np.cos(ai), cy + ri*np.sin(ai)) for ri, ai in zip(r, angles)]
    elif method == 'smooth':
        t = np.linspace(0, 2*np.pi, 100, endpoint=False)
        r = 0.035 + sum(np.random.uniform(-0.008, 0.008) * np.cos(k*t + np.random.uniform(0, 2*np.pi)) for k in range(2, 8))
        pts_c = [(cx + ri*np.cos(ti), cy + ri*np.sin(ti)) for ri, ti in zip(r, t)]
    elif method == 'pillbox':
        body_w = np.random.uniform(0.05, 0.09)
        body_h = np.random.uniform(0.04, 0.07)
        pipe_w = np.random.uniform(0.015, body_w - 0.01)
        pipe_top = np.random.uniform(0.01, 0.025)
        pipe_bot = np.random.uniform(0.01, 0.025)
        
        x_b_min, x_b_max = cx - body_w/2, cx + body_w/2
        y_b_min, y_b_max = cy - body_h/2, cy + body_h/2
        x_p_min, x_p_max = cx - pipe_w/2, cx + pipe_w/2
        
        pts_c = [
            (x_p_min, y_b_max + pipe_top), (x_p_min, y_b_max), 
            (x_b_min, y_b_max), (x_b_min, y_b_min), (x_p_min, y_b_min), 
            (x_p_min, y_b_min - pipe_bot), (x_p_max, y_b_min - pipe_bot), 
            (x_p_max, y_b_min), (x_b_max, y_b_min), (x_b_max, y_b_max), 
            (x_p_max, y_b_max), (x_p_max, y_b_max + pipe_top)
        ]
    else: # elliptical (TESLA-like)
        req = np.random.uniform(0.040, 0.055)    # Equator radius
        riris = np.random.uniform(0.012, 0.025)  # Beam pipe / Iris radius
        L = np.random.uniform(0.04, 0.08)        # Cell length
        L_pipe = np.random.uniform(0.01, 0.03)   # Beam pipe length
        power = np.random.uniform(1.5, 3.0)      # Curvature control (2.0 is harmonic)
        
        # We need a CCW boundary. 
        # Start Top-Right -> Top-Curve -> Top-Left -> Bot-Left -> Bot-Curve -> Bot-Right -> Top-Right
        x_curve = np.linspace(L/2, -L/2, 40) # Right to Left
        y_top_curve = cy + riris + (req - riris) * (np.cos(x_curve * np.pi / L)**power)
        
        pts_c = []
        # Top-Right pipe
        pts_c.append((cx + L/2 + L_pipe, cy + riris))
        pts_c.append((cx + L/2, cy + riris))
        # Top Cell Curve (Right to Left)
        for x, y in zip(x_curve, y_top_curve):
            pts_c.append((cx + x, y))
        # Top-Left pipe
        pts_c.append((cx - L/2, cy + riris))
        pts_c.append((cx - L/2 - L_pipe, cy + riris))
        
        # Bottom-Left pipe
        pts_c.append((cx - L/2 - L_pipe, cy - riris))
        pts_c.append((cx - L/2, cy - riris))
        # Bottom Cell Curve (Left to Right)
        x_curve_bot = np.linspace(-L/2, L/2, 40)
        y_bot_curve = cy - (riris + (req - riris) * (np.cos(x_curve_bot * np.pi / L)**power))
        for x, y in zip(x_curve_bot, y_bot_curve):
            pts_c.append((cx + x, y))
        # Bottom-Right pipe
        pts_c.append((cx + L/2, cy - riris))
        pts_c.append((cx + L/2 + L_pipe, cy - riris))
    
    # Close the boundary for plotting
    pts_c.append(pts_c[0])
    return np.array(pts_c)

def main():
    methods = ['pillbox', 'elliptical', 'smooth', 'sharp']
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            method = methods[i]
            pts = generate_geometry(method)
            ax.plot(pts[:, 0], pts[:, 1], '-b', linewidth=2)
            ax.fill(pts[:, 0], pts[:, 1], alpha=0.2, color='blue')
            ax.set_aspect('equal')
            ax.set_title(method.capitalize())
            ax.axis('off')
            
    plt.tight_layout()
    output_path = "cavity_shapes_demo.png"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Shapes saved to {output_path}")

if __name__ == '__main__':
    main()
