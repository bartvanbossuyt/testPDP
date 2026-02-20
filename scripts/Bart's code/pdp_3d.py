import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go


G = 9.81
TARGET_MAX_HEIGHT = 30.0


def build_trajectory(
	azimuth_deg: float,
	elevation_deg: float,
	n_timestamps: int,
	drag_coefficient: float,
	wind_d1: float,
	wind_d2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Build trajectory with gravity + quadratic air drag using wind-relative velocity."""
	elevation_rad = np.deg2rad(elevation_deg)
	azimuth_rad = np.deg2rad(azimuth_deg)

	# Keep original launch setup: in still air this reaches ~30 m.
	vz0 = float(np.sqrt(2.0 * G * TARGET_MAX_HEIGHT))
	v0 = float(vz0 / np.sin(elevation_rad))
	vxy = float(v0 * np.cos(elevation_rad))

	vx0 = float(vxy * np.cos(azimuth_rad))
	vy0 = float(vxy * np.sin(azimuth_rad))

	# Numerical integration (semi-implicit Euler)
	dt = 0.01
	time_max = 20.0

	t_hist = [0.0]
	x_hist = [0.0]
	y_hist = [0.0]
	z_hist = [0.0]

	vel = np.array([vx0, vy0, vz0], dtype=float)
	pos = np.array([0.0, 0.0, 0.0], dtype=float)
	wind = np.array([wind_d1, wind_d2, 0.0], dtype=float)

	t_curr = 0.0
	while t_curr < time_max:
		v_rel = vel - wind
		speed_rel = float(np.linalg.norm(v_rel))
		drag_acc = -drag_coefficient * speed_rel * v_rel
		gravity_acc = np.array([0.0, 0.0, -G], dtype=float)
		acc = gravity_acc + drag_acc

		vel = vel + acc * dt
		pos = pos + vel * dt
		t_curr += dt

		if pos[2] <= 0.0 and t_curr > 0.02:
			pos[2] = 0.0
			t_hist.append(t_curr)
			x_hist.append(float(pos[0]))
			y_hist.append(float(pos[1]))
			z_hist.append(0.0)
			break

		t_hist.append(t_curr)
		x_hist.append(float(pos[0]))
		y_hist.append(float(pos[1]))
		z_hist.append(float(max(pos[2], 0.0)))

	# Resample to exactly n_timestamps for matrix construction and slider.
	t_dense = np.array(t_hist, dtype=float)
	x_dense = np.array(x_hist, dtype=float)
	y_dense = np.array(y_hist, dtype=float)
	z_dense = np.array(z_hist, dtype=float)

	t = np.linspace(float(t_dense[0]), float(t_dense[-1]), int(n_timestamps))
	x = np.interp(t, t_dense, x_dense)
	y = np.interp(t, t_dense, y_dense)
	z = np.interp(t, t_dense, z_dense)
	z[-1] = 0.0

	return t, x, y, z


def inequality_matrix(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
	"""Return an NxN matrix using inverse.py-style ordering codes.

	0 -> lower-than, 1 -> equal, 2 -> greater-than
	"""
	n = len(values)
	matrix = np.zeros((n, n), dtype=int)
	for i in range(n):
		for j in range(n):
			delta = values[i] - values[j]
			if delta > eps:
				matrix[i, j] = 2
			elif delta < -eps:
				matrix[i, j] = 0
			else:
				matrix[i, j] = 1
	return matrix


def create_heatmap_figure(matrix: np.ndarray, title: str) -> plt.Figure:
	"""Create an inverse.py-like heatmap for an inequality matrix."""
	n = matrix.shape[0]
	labels = [f"t{i}" for i in range(n)]
	cmap = ListedColormap(["#00AA00", "#FFEB3B", "#E53935"])

	fig, ax = plt.subplots(figsize=(3.4, 3.4))
	ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="equal")
	ax.set_title(title, fontsize=10, fontweight="bold")

	if n <= 20:
		ax.set_xticks(range(n))
		ax.set_yticks(range(n))
		ax.set_xticklabels(labels, fontsize=7, rotation=90)
		ax.set_yticklabels(labels, fontsize=7)
	else:
		ax.set_xticks([])
		ax.set_yticks([])

	fig.tight_layout()
	return fig


st.set_page_config(page_title="3D Ball Motion + PDP Matrices", layout="wide")
st.title("3D Ball Motion (Parabola) with d1/d2/d3 Inequality Matrices")
st.caption("d1 = x-position, d2 = y-position, d3 = height (z). Includes gravity, wind, and air friction. Ball stops when it hits the ground.")


ctrl1, ctrl2, ctrl3 = st.columns(3)
with ctrl1:
	azimuth_deg = st.slider(
		"Initial launch direction in x-y plane (azimuth °)",
		min_value=-180,
		max_value=180,
		value=35,
		step=1,
		help="0° launches along +d1. Positive angles rotate toward +d2.",
	)
with ctrl2:
	elevation_deg = st.slider(
		"Initial elevation angle (°)",
		min_value=10,
		max_value=80,
		value=50,
		step=1,
		help="Direction above the ground plane. Peak height stays fixed at 30.",
	)
with ctrl3:
	n_timestamps = st.slider(
		"Number of timestamps",
		min_value=6,
		max_value=20,
		value=10,
		step=1,
	)

wind_col1, wind_col2, wind_col3 = st.columns(3)
with wind_col1:
	drag_coefficient = st.slider(
		"Air friction strength",
		min_value=0.0,
		max_value=0.08,
		value=0.02,
		step=0.001,
		help="Quadratic drag strength. Higher = stronger slowdown.",
	)
with wind_col2:
	wind_d1 = st.slider(
		"Wind speed d1 (m/s)",
		min_value=-20.0,
		max_value=20.0,
		value=0.0,
		step=0.5,
		help="Positive wind pushes toward +d1.",
	)
with wind_col3:
	wind_d2 = st.slider(
		"Wind speed d2 (m/s)",
		min_value=-20.0,
		max_value=20.0,
		value=0.0,
		step=0.5,
		help="Positive wind pushes toward +d2.",
	)

t, x, y, z = build_trajectory(
	azimuth_deg=azimuth_deg,
	elevation_deg=elevation_deg,
	n_timestamps=n_timestamps,
	drag_coefficient=drag_coefficient,
	wind_d1=wind_d1,
	wind_d2=wind_d2,
)

idx = st.slider(
	"Motion step",
	min_value=0,
	max_value=n_timestamps - 1,
	value=0,
	step=1,
	help="Shows the ball position at the selected timestamp.",
)

left, right = st.columns([2, 1])
with left:
	max_range = float(max(np.max(np.abs(x)), np.max(np.abs(y)), TARGET_MAX_HEIGHT))
	x_min, x_max = -0.1 * max_range, max_range * 1.05
	y_min, y_max = -0.55 * max_range, 0.55 * max_range
	z_min, z_max = 0.0, TARGET_MAX_HEIGHT * 1.1

	fig3d = go.Figure()
	fig3d.add_trace(
		go.Scatter3d(
			x=x,
			y=y,
			z=z,
			mode="lines",
			name="Trajectory",
			line=dict(width=6),
		)
	)
	fig3d.add_trace(
		go.Scatter3d(
			x=[x[idx]],
			y=[y[idx]],
			z=[z[idx]],
			mode="markers",
			name="Ball",
			marker=dict(size=7),
		)
	)
	fig3d.add_trace(
		go.Scatter3d(
			x=[x[0], x[-1]],
			y=[y[0], y[-1]],
			z=[z[0], z[-1]],
			mode="markers",
			name="Start/End",
			marker=dict(size=5),
		)
	)

	fig3d.update_layout(
		title="3D trajectory and current ball position (drag to rotate)",
		height=560,
		margin=dict(l=0, r=0, b=0, t=40),
		scene=dict(
			xaxis=dict(title="d1 (x)", range=[x_min, x_max]),
			yaxis=dict(title="d2 (y)", range=[y_min, y_max]),
			zaxis=dict(title="d3 (height)", range=[z_min, z_max]),
			aspectmode="cube",
		),
	)

	st.plotly_chart(fig3d, use_container_width=True)

with right:
	st.markdown("### Current state")
	st.write(f"t = {t[idx]:.2f} s")
	st.write(f"d1 = {x[idx]:.2f}")
	st.write(f"d2 = {y[idx]:.2f}")
	st.write(f"d3 = {z[idx]:.2f}")
	st.write(f"Max d3 = {np.max(z):.2f}")
	st.write(f"Ground hit at t = {t[-1]:.2f} s")


st.markdown("---")
st.markdown("### PDP Inequality Matrices")
st.caption("Cell values use inverse-style coding: 0 = lower-than (green), 1 = equal (yellow), 2 = greater-than (red).")

matrix_d1 = inequality_matrix(x)
matrix_d2 = inequality_matrix(y)
matrix_d3 = inequality_matrix(z)

hm1, hm2, hm3 = st.columns(3)
with hm1:
	st.markdown("**d1 matrix**")
	fig_d1 = create_heatmap_figure(matrix_d1, "d1 (x)")
	st.pyplot(fig_d1)
	plt.close(fig_d1)

with hm2:
	st.markdown("**d2 matrix**")
	fig_d2 = create_heatmap_figure(matrix_d2, "d2 (y)")
	st.pyplot(fig_d2)
	plt.close(fig_d2)

with hm3:
	st.markdown("**d3 matrix**")
	fig_d3 = create_heatmap_figure(matrix_d3, "d3 (height)")
	st.pyplot(fig_d3)
	plt.close(fig_d3)
