"""pygame 弹窗逐帧显示 HWC uint8 画面。

pygame 延迟 import：无头/CI 环境不触发。渲染只做显示，不参与数值链路。
"""

RENDER_SCALE = 4


class PyGameViewer:
    def __init__(self, size_hw, title):
        import pygame

        self._dim = (int(size_hw[1] * RENDER_SCALE), int(size_hw[0] * RENDER_SCALE))
        pygame.display.set_caption(title)
        self._screen = pygame.display.set_mode(self._dim)

    def show(self, frame_hwc_uint8):
        import pygame
        import pygame.surfarray

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit("window closed")
        surface = pygame.surfarray.make_surface(frame_hwc_uint8.swapaxes(0, 1))
        scaled = pygame.transform.smoothscale(surface, self._dim)
        self._screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def close(self):
        import pygame

        pygame.display.quit()
        pygame.quit()
