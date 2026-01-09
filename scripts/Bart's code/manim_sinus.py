"""
3D Manim animatie van een bewegende sinusgolf
Run met: python -m manim -pql manim_sinus.py MovingSineWave3D
Of voor hoge kwaliteit: python -m manim -pqh manim_sinus.py MovingSineWave3D
"""

from manim import *
import numpy as np


class MovingSineWave3D(ThreeDScene):
    """3D animatie van een sinusgolf die door de ruimte beweegt."""
    
    def construct(self):
        # Camera instellen voor 3D view
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        
        # Assen toevoegen
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-2, 2, 1],
            z_range=[-2, 2, 1],
            x_length=8,
            y_length=4,
            z_length=4,
        )
        
        # Toon assen (zonder labels - vereist LaTeX)
        self.play(Create(axes))
        
        # Initiële sinusgolf
        t_tracker = ValueTracker(0)
        
        def get_sine_wave():
            t = t_tracker.get_value()
            return ParametricFunction(
                lambda x: axes.c2p(
                    x,
                    np.sin(2 * x - t),  # Golf beweegt in x-richting
                    np.cos(2 * x - t) * 0.5  # 3D spiraal effect
                ),
                t_range=[-4, 4],
                color=BLUE,
                stroke_width=4,
            )
        
        sine_wave = always_redraw(get_sine_wave)
        
        # Toon de golf
        self.play(Create(sine_wave))
        
        # Camera langzaam laten roteren
        self.begin_ambient_camera_rotation(rate=0.2)
        
        # Animeer de golf
        self.play(
            t_tracker.animate.set_value(4 * PI),
            run_time=8,
            rate_func=linear
        )
        
        # Stop camera rotatie
        self.stop_ambient_camera_rotation()
        
        # Wacht even
        self.wait()


class SineWaveWithTrail3D(ThreeDScene):
    """3D sinusgolf met trail effect - meerdere golven op verschillende hoogtes."""
    
    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-60 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 2, 1],
            x_length=8,
            y_length=6,
            z_length=4,
        )
        
        self.add(axes)
        
        t_tracker = ValueTracker(0)
        
        # Meerdere golven op verschillende y-posities
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
        waves = VGroup()
        
        for i, color in enumerate(colors):
            y_offset = -2.5 + i * 1.0
            phase_offset = i * PI / 6
            
            def make_wave(y_off=y_offset, phase=phase_offset, col=color):
                def get_wave():
                    t = t_tracker.get_value()
                    return ParametricFunction(
                        lambda x, y=y_off, p=phase: axes.c2p(
                            x,
                            y,
                            np.sin(2 * x - t + p)
                        ),
                        t_range=[-4, 4],
                        color=col,
                        stroke_width=3,
                    )
                return always_redraw(get_wave)
            
            waves.add(make_wave())
        
        self.play(Create(waves), run_time=2)
        
        self.begin_ambient_camera_rotation(rate=0.15)
        
        self.play(
            t_tracker.animate.set_value(4 * PI),
            run_time=10,
            rate_func=linear
        )
        
        self.stop_ambient_camera_rotation()
        self.wait()


class SineSurface3D(ThreeDScene):
    """3D oppervlak gebaseerd op sinusgolven."""
    
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-2, 2, 1],
            x_length=8,
            y_length=8,
            z_length=4,
        )
        
        self.add(axes)
        
        t_tracker = ValueTracker(0)
        
        def get_surface():
            t = t_tracker.get_value()
            return Surface(
                lambda u, v: axes.c2p(
                    u,
                    v,
                    np.sin(u - t) * np.cos(v - t * 0.5) * 0.8
                ),
                u_range=[-3, 3],
                v_range=[-3, 3],
                resolution=(30, 30),
                fill_opacity=0.7,
                checkerboard_colors=[BLUE_D, BLUE_E],
                stroke_width=0.5,
                stroke_color=WHITE,
            )
        
        surface = always_redraw(get_surface)
        
        self.play(Create(surface), run_time=2)
        
        self.begin_ambient_camera_rotation(rate=0.1)
        
        self.play(
            t_tracker.animate.set_value(2 * PI),
            run_time=8,
            rate_func=linear
        )
        
        self.stop_ambient_camera_rotation()
        self.wait()


if __name__ == "__main__":
    # Dit bestand moet gerund worden met manim CLI:
    # manim -pql manim_sinus.py MovingSineWave3D
    # manim -pql manim_sinus.py SineWaveWithTrail3D
    # manim -pql manim_sinus.py SineSurface3D
    print("Run dit bestand met:")
    print("  manim -pql manim_sinus.py MovingSineWave3D")
    print("  manim -pql manim_sinus.py SineWaveWithTrail3D")
    print("  manim -pql manim_sinus.py SineSurface3D")
    print("\nVoor hoge kwaliteit, gebruik -pqh in plaats van -pql")
