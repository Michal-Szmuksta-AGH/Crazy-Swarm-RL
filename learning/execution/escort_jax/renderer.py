from __future__ import annotations

import numpy as np


class Pygame3DRenderer:
    """Realtime pygame+OpenGL renderer with sphere-based drones/targets."""

    def __init__(
        self,
        window_size: int = 1400,
        map_size: int = 2,
        fps: int = 20,
        camera_distance: float = 5.5,
        camera_yaw_deg: float = -70.0,
        camera_pitch_deg: float = 30.0,
    ):
        import pygame
        from pygame import DOUBLEBUF, OPENGL
        from OpenGL.GL import (
            GL_AMBIENT,
            GL_AMBIENT_AND_DIFFUSE,
            GL_BLEND,
            GL_COLOR_BUFFER_BIT,
            GL_COLOR_MATERIAL,
            GL_DEPTH_BUFFER_BIT,
            GL_DEPTH_TEST,
            GL_DIFFUSE,
            GL_FRONT_AND_BACK,
            GL_LIGHT0,
            GL_LIGHTING,
            GL_MODELVIEW,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_POSITION,
            GL_PROJECTION,
            GL_SMOOTH,
            GL_SRC_ALPHA,
            glBlendFunc,
            glClear,
            glColor4f,
            glColorMaterial,
            glEnable,
            glLight,
            glLightfv,
            glLineWidth,
            glLoadIdentity,
            glMatrixMode,
            glShadeModel,
        )
        from OpenGL.raw.GLU import gluLookAt, gluPerspective

        from crazy_rl.utils.graphic import axes, field, point, target_point

        self._pygame = pygame
        self._draw_axes = axes
        self._draw_field = field
        self._draw_point = point
        self._draw_target = target_point

        self._gl = {
            "glEnable": glEnable,
            "glShadeModel": glShadeModel,
            "glColorMaterial": glColorMaterial,
            "glBlendFunc": glBlendFunc,
            "glLightfv": glLightfv,
            "glLight": glLight,
            "glMatrixMode": glMatrixMode,
            "glLoadIdentity": glLoadIdentity,
            "gluPerspective": gluPerspective,
            "gluLookAt": gluLookAt,
            "glClear": glClear,
            "glColor4f": glColor4f,
            "glLineWidth": glLineWidth,
        }
        self._const = {
            "DOUBLEBUF": DOUBLEBUF,
            "OPENGL": OPENGL,
            "GL_DEPTH_TEST": GL_DEPTH_TEST,
            "GL_LIGHTING": GL_LIGHTING,
            "GL_SMOOTH": GL_SMOOTH,
            "GL_COLOR_MATERIAL": GL_COLOR_MATERIAL,
            "GL_FRONT_AND_BACK": GL_FRONT_AND_BACK,
            "GL_AMBIENT_AND_DIFFUSE": GL_AMBIENT_AND_DIFFUSE,
            "GL_SRC_ALPHA": GL_SRC_ALPHA,
            "GL_ONE_MINUS_SRC_ALPHA": GL_ONE_MINUS_SRC_ALPHA,
            "GL_BLEND": GL_BLEND,
            "GL_LIGHT0": GL_LIGHT0,
            "GL_AMBIENT": GL_AMBIENT,
            "GL_DIFFUSE": GL_DIFFUSE,
            "GL_PROJECTION": GL_PROJECTION,
            "GL_MODELVIEW": GL_MODELVIEW,
            "GL_COLOR_BUFFER_BIT": GL_COLOR_BUFFER_BIT,
            "GL_DEPTH_BUFFER_BIT": GL_DEPTH_BUFFER_BIT,
            "GL_POSITION": GL_POSITION,
        }

        self.window_size = int(window_size)
        self.map_size = int(map_size)
        self.fps = int(fps)
        self.camera_target = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        self.camera_distance = float(camera_distance)
        self.camera_yaw_deg = float(camera_yaw_deg)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self._dragging = False
        self._last_mouse_xy = None

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Escort JAX Evaluation (3D)")
        self.window = pygame.display.set_mode((self.window_size, self.window_size), DOUBLEBUF | OPENGL)
        self.clock = pygame.time.Clock()
        self.closed = False

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glLineWidth(1.5)

        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

        glMatrixMode(GL_PROJECTION)
        gluPerspective(70, 1.0, 0.1, 60.0)
        glMatrixMode(GL_MODELVIEW)

    def _handle_event(self, event) -> bool:
        if event.type == self._pygame.QUIT:
            self.closed = True
            return False

        if event.type == self._pygame.KEYDOWN:
            if event.key in (self._pygame.K_ESCAPE, self._pygame.K_q):
                self.closed = True
                return False
            if event.key == self._pygame.K_r:
                self.camera_target = np.array([0.0, 0.0, 0.8], dtype=np.float32)
                self.camera_distance = 5.5
                self.camera_yaw_deg = -70.0
                self.camera_pitch_deg = 30.0

        if event.type == self._pygame.MOUSEBUTTONDOWN:
            if event.button == 3:
                self._dragging = True
                self._last_mouse_xy = np.array(event.pos, dtype=np.float32)
            elif event.button == 4:
                self.camera_distance = max(2.0, self.camera_distance - 0.30)
            elif event.button == 5:
                self.camera_distance = min(30.0, self.camera_distance + 0.30)

        if event.type == self._pygame.MOUSEBUTTONUP and event.button == 3:
            self._dragging = False
            self._last_mouse_xy = None

        if event.type == self._pygame.MOUSEMOTION and self._dragging and self._last_mouse_xy is not None:
            current_xy = np.array(event.pos, dtype=np.float32)
            delta_xy = current_xy - self._last_mouse_xy
            self._last_mouse_xy = current_xy
            self.camera_yaw_deg += float(delta_xy[0]) * 0.25
            self.camera_pitch_deg = float(np.clip(self.camera_pitch_deg - delta_xy[1] * 0.20, -85.0, 85.0))

        return True

    def _update_camera_target_from_keyboard(self) -> None:
        keys = self._pygame.key.get_pressed()
        step_xy = 0.06
        step_z = 0.04
        if keys[self._pygame.K_w]:
            self.camera_target[1] += step_xy
        if keys[self._pygame.K_s]:
            self.camera_target[1] -= step_xy
        if keys[self._pygame.K_a]:
            self.camera_target[0] -= step_xy
        if keys[self._pygame.K_d]:
            self.camera_target[0] += step_xy
        if keys[self._pygame.K_e]:
            self.camera_target[2] += step_z
        if keys[self._pygame.K_c]:
            self.camera_target[2] -= step_z

    def _apply_camera(self) -> None:
        yaw = np.deg2rad(self.camera_yaw_deg)
        pitch = np.deg2rad(self.camera_pitch_deg)
        cx = self.camera_target[0] + self.camera_distance * np.cos(pitch) * np.cos(yaw)
        cy = self.camera_target[1] + self.camera_distance * np.cos(pitch) * np.sin(yaw)
        cz = self.camera_target[2] + self.camera_distance * np.sin(pitch)

        self._gl["gluLookAt"](
            float(cx),
            float(cy),
            float(cz),
            float(self.camera_target[0]),
            float(self.camera_target[1]),
            float(self.camera_target[2]),
            0.0,
            0.0,
            1.0,
        )

    @staticmethod
    def _world_to_gl(point: tuple[float, float, float]) -> tuple[float, float, float]:
        """Map world (x, y, z) to the OpenGL coordinates used by graphic.py helpers."""
        x, y, z = point
        return (-y, x, z - 2.0)

    def _draw_environment_frame(self) -> None:
        """Draw wireframe box showing environment boundaries."""
        from OpenGL.GL import glBegin, glEnd, glVertex3f, GL_LINES
        s = self.map_size
        h = self.map_size  # Use map_size for consistent cubic bounds
        self._gl["glColor4f"](0.3, 0.3, 0.5, 1.0)
        self._gl["glLineWidth"](2.0)
        glBegin(GL_LINES)
        # Bottom edges (z=0)
        corners_bottom = [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)]
        for i in range(4):
            p1 = self._world_to_gl(corners_bottom[i])
            p2 = self._world_to_gl(corners_bottom[(i + 1) % 4])
            glVertex3f(*p1)
            glVertex3f(*p2)
        # Top edges (z=h)
        corners_top = [(-s, -s, float(h)), (s, -s, float(h)), (s, s, float(h)), (-s, s, float(h))]
        for i in range(4):
            p1 = self._world_to_gl(corners_top[i])
            p2 = self._world_to_gl(corners_top[(i + 1) % 4])
            glVertex3f(*p1)
            glVertex3f(*p2)
        # Vertical edges
        for i in range(4):
            glVertex3f(*self._world_to_gl(corners_bottom[i]))
            glVertex3f(*self._world_to_gl(corners_top[i]))
        glEnd()
        self._gl["glLineWidth"](1.5)

    def render(self, agent_xyz: np.ndarray, target_xyz: np.ndarray, title: str = "") -> bool:
        from OpenGL.GL import glPopMatrix, glPushMatrix

        for event in self._pygame.event.get():
            if not self._handle_event(event):
                return False

        self._update_camera_target_from_keyboard()
        
        # Add target coordinates to the title
        if len(target_xyz) > 0:
            target_str = f"Target: ({target_xyz[0, 0]:.2f}, {target_xyz[0, 1]:.2f}, {target_xyz[0, 2]:.2f})"
            if title:
                full_title = f"{title} | {target_str}"
            else:
                full_title = f"Escort JAX Evaluation (3D) | {target_str}"
        else:
            full_title = title or "Escort JAX Evaluation (3D)"
        
        self._pygame.display.set_caption(full_title)

        self._gl["glLoadIdentity"]()
        self._apply_camera()
        self._gl["glLight"](self._const["GL_LIGHT0"], self._const["GL_POSITION"], (-1, -1, 5, 1))
        self._gl["glClear"](self._const["GL_COLOR_BUFFER_BIT"] | self._const["GL_DEPTH_BUFFER_BIT"])

        self._gl["glColor4f"](0.5, 0.5, 0.5, 1)
        self._draw_field(self.map_size)
        self._draw_axes()
        self._draw_environment_frame()

        for p in agent_xyz:
            glPushMatrix()
            self._draw_point(np.array([p[0], p[1], p[2]], dtype=np.float32))
            glPopMatrix()

        for t in target_xyz:
            glPushMatrix()
            self._draw_target(np.array([t[0], t[1], t[2]], dtype=np.float32))
            glPopMatrix()

        self._pygame.display.flip()
        self.clock.tick(max(1, self.fps))
        return True

    def close(self) -> None:
        if not self.closed:
            self._pygame.display.quit()
            self._pygame.quit()
            self.closed = True
