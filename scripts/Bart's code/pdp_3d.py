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
	air_density: float,
	drag_coefficient: float,
	ball_radius: float,
	ball_mass: float,
	wind_d1: float,
	wind_d2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
	"""Build trajectory with physical drag model and launch speed solved for ~30 m apex."""
	elevation_rad = np.deg2rad(elevation_deg)
	azimuth_rad = np.deg2rad(azimuth_deg)

	area = float(np.pi * ball_radius**2)
	k_drag = float(0.5 * air_density * drag_coefficient * area / max(ball_mass, 1e-6))

	def simulate_with_speed(v0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		vxy = float(v0 * np.cos(elevation_rad))
		vz0 = float(v0 * np.sin(elevation_rad))
		vx0 = float(vxy * np.cos(azimuth_rad))
		vy0 = float(vxy * np.sin(azimuth_rad))

		dt = 0.005
		time_max = 40.0

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
			drag_acc = -k_drag * speed_rel * v_rel
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

		return (
			np.array(t_hist, dtype=float),
			np.array(x_hist, dtype=float),
			np.array(y_hist, dtype=float),
			np.array(z_hist, dtype=float),
		)

	# Solve launch speed so max height is ~TARGET_MAX_HEIGHT, also with drag.
	v_no_drag = float(np.sqrt(2.0 * G * TARGET_MAX_HEIGHT) / np.sin(elevation_rad))
	low_v = 0.2 * v_no_drag
	high_v = max(1.5 * v_no_drag, 1.0)

	def max_height_for_speed(v: float) -> float:
		_, _, _, z_tmp = simulate_with_speed(v)
		return float(np.max(z_tmp))

	while max_height_for_speed(high_v) < TARGET_MAX_HEIGHT and high_v < 500.0:
		high_v *= 1.5

	for _ in range(28):
		mid = 0.5 * (low_v + high_v)
		if max_height_for_speed(mid) < TARGET_MAX_HEIGHT:
			low_v = mid
		else:
			high_v = mid

	v0_solved = high_v
	t_dense, x_dense, y_dense, z_dense = simulate_with_speed(v0_solved)

	# Resample to exactly n_timestamps for matrix construction and slider.
	t = np.linspace(float(t_dense[0]), float(t_dense[-1]), int(n_timestamps))
	x = np.interp(t, t_dense, x_dense)
	y = np.interp(t, t_dense, y_dense)
	z = np.interp(t, t_dense, z_dense)
	z[-1] = 0.0

	return t, x, y, z, v0_solved


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
		"Drag coefficient Cd",
		min_value=0.05,
		max_value=1.2,
		value=0.47,
		step=0.01,
		help="Aerodynamic drag coefficient. Sphere-like balls are often around 0.4-0.5.",
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

phys1, phys2, phys3 = st.columns(3)
with phys1:
	air_density = st.slider(
		"Air density ρ (kg/m³)",
		min_value=0.8,
		max_value=1.4,
		value=1.225,
		step=0.005,
	)
with phys2:
	ball_radius = st.slider(
		"Ball radius (m)",
		min_value=0.02,
		max_value=0.15,
		value=0.033,
		step=0.001,
	)
with phys3:
	ball_mass = st.slider(
		"Ball mass (kg)",
		min_value=0.02,
		max_value=1.0,
		value=0.058,
		step=0.001,
	)

t, x, y, z, launch_speed = build_trajectory(
	azimuth_deg=azimuth_deg,
	elevation_deg=elevation_deg,
	n_timestamps=n_timestamps,
	air_density=air_density,
	drag_coefficient=drag_coefficient,
	ball_radius=ball_radius,
	ball_mass=ball_mass,
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
	x_data_min, x_data_max = float(np.min(x)), float(np.max(x))
	y_data_min, y_data_max = float(np.min(y)), float(np.max(y))

	x_span = max(1.0, x_data_max - x_data_min)
	y_span = max(1.0, y_data_max - y_data_min)

	# Larger margins to keep the full trajectory visible under strong wind/drag.
	x_margin = 0.25 * x_span
	y_margin = 0.25 * y_span

	x_min, x_max = x_data_min - x_margin, x_data_max + x_margin
	y_min, y_max = y_data_min - y_margin, y_data_max + y_margin
	z_min, z_max = 0.0, TARGET_MAX_HEIGHT * 1.1

	wind_mag = float(np.hypot(wind_d1, wind_d2))
	wind_dir_deg = float(np.degrees(np.arctan2(wind_d2, wind_d1))) if wind_mag > 1e-9 else 0.0

	origin_x, origin_y = 0.08, 0.92
	if wind_mag > 1e-9:
		u_x = wind_d1 / wind_mag
		u_y = wind_d2 / wind_mag
	else:
		u_x, u_y = 1.0, 0.0

	arrow_len = 0.07 + min(0.12, wind_mag * 0.006)
	end_x = origin_x + arrow_len * u_x
	end_y = origin_y + arrow_len * u_y

	head_len = 0.022
	head_angle = np.deg2rad(26.0)
	back_x, back_y = -u_x, -u_y
	lx = back_x * np.cos(head_angle) - back_y * np.sin(head_angle)
	ly = back_x * np.sin(head_angle) + back_y * np.cos(head_angle)
	rx = back_x * np.cos(-head_angle) - back_y * np.sin(-head_angle)
	ry_ = back_x * np.sin(-head_angle) + back_y * np.cos(-head_angle)

	left_head_x = end_x + head_len * lx
	left_head_y = end_y + head_len * ly
	right_head_x = end_x + head_len * rx
	right_head_y = end_y + head_len * ry_

	wind_shapes = [
		dict(
			type="line",
			xref="paper",
			yref="paper",
			x0=origin_x,
			y0=origin_y,
			x1=end_x,
			y1=end_y,
			line=dict(color="black", width=1),
		),
		dict(
			type="line",
			xref="paper",
			yref="paper",
			x0=end_x,
			y0=end_y,
			x1=left_head_x,
			y1=left_head_y,
			line=dict(color="black", width=1),
		),
		dict(
			type="line",
			xref="paper",
			yref="paper",
			x0=end_x,
			y0=end_y,
			x1=right_head_x,
			y1=right_head_y,
			line=dict(color="black", width=1),
		),
	]

	wind_annotations = [
		dict(
			xref="paper",
			yref="paper",
			x=origin_x,
			y=origin_y - 0.045,
			text=f"Wind: {wind_mag:.1f} m/s | {wind_dir_deg:.0f}°",
			showarrow=False,
			font=dict(size=11, color="black"),
			align="left",
		)
	]

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
			x=[x[0]],
			y=[y[0]],
			z=[z[0]],
			mode="markers",
			name="Start",
			marker=dict(size=6, color="lime"),
		)
	)
	fig3d.add_trace(
		go.Scatter3d(
			x=[x[-1]],
			y=[y[-1]],
			z=[z[-1]],
			mode="markers",
			name="End",
			marker=dict(size=6),
		)
	)

	fig3d.update_layout(
		title="3D trajectory and current ball position (drag to rotate)",
		height=560,
		margin=dict(l=0, r=0, b=0, t=40),
		shapes=wind_shapes,
		annotations=wind_annotations,
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
	st.write(f"Launch speed solved for target apex ≈ 30 m: {launch_speed:.2f} m/s")


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
