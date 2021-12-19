from copy import deepcopy
import multiprocessing
import sys
import tempfile
import threading

import pdf2image
import pyglet
import tqdm


def bestmode(screen):
    return max(screen.get_modes(), key=lambda m: m.height)


REPEAT_TRIGGER = 0.4
REPEAT_INTERVAL = 0.1

UP = 0
HOLD = 1
FIRE = 2

KEYS_FWD = [
    pyglet.window.key.RIGHT,
    pyglet.window.key.UP,
    pyglet.window.key.PAGEDOWN,
]
KEYS_REV = [
    pyglet.window.key.LEFT,
    pyglet.window.key.DOWN,
    pyglet.window.key.PAGEUP,
]

class Repeater:
    """Implements repeat-after-hold, similar to OS keyboard repeating."""
    def __init__(self):
        self.state = UP
        # TODO: should never read uninitialized...
        self.stopwatch = -1000000000000

    def tick(self, dt, is_down):
        """Processes one time interval and returns the number of repeats fired.

        Args:
            dt: Time interval in seconds.
            is_down: State of the key/button during interval.

        Returns: The number of repeats fired during the interval.
        """
        if not is_down:
            self.state = UP
            return 0
        # Key is down.
        if self.state == UP:
            self.state = HOLD
            self.stopwatch = dt
            # Rising edge fire.
            return 1
        elif self.state == HOLD:
            self.stopwatch += dt
            if self.stopwatch < REPEAT_TRIGGER:
                return 0
            else:
                self.state = FIRE
                self.stopwatch -= REPEAT_TRIGGER
                return 1 + self._countdown()
        elif self.state == FIRE:
            self.stopwatch += dt
            return self._countdown()

    def _countdown(self):
        fires = 0
        while self.stopwatch > REPEAT_INTERVAL:
            fires += 1
            self.stopwatch -= REPEAT_INTERVAL
        return fires


class Cursor:
    """Implements cursor logic."""
    def __init__(self, nslides):
        self.rev = Repeater()
        self.fwd = Repeater()
        self.cursor = 0
        self.nslides = nslides

    def tick(self, dt, reverse, forward):
        """Returns True if the cursor changed, false otherwise."""
        old_value = self.cursor
        # TODO: Make sure this is the right thing to do when both are held.
        if reverse and forward:
            return False
        self.cursor -= self.rev.tick(dt, reverse)
        self.cursor += self.fwd.tick(dt, forward)
        self.cursor = min(self.cursor, self.nslides - 1)
        self.cursor = max(self.cursor, 0)
        return self.cursor != old_value


def rasterize(pdfpath, width, height, progressbar=True, pagelimit=None):
    with tempfile.TemporaryDirectory() as tempdir:
        paths = pdf2image.convert_from_path(
            pdfpath,
            size=(width, height),
            output_folder=tempdir,
            # Do not bother loading as PIL images. Let Pyglet handle loading.
            # TODO: Try to keep everything in memory.
            last_page=pagelimit,
            paths_only=True,
            thread_count=4,
        )
        if progressbar:
            paths = tqdm.tqdm(paths)
        # TODO: Why is this so slow?
        imgs = [pyglet.image.load(p) for p in paths]
        return imgs


def winsize2rasterargs(window_size, aspect):
    width, height = window_size
    window_aspect = float(width) / height
    if window_aspect >= aspect:
        width = None
    else:
        height = None
    return (width, height)


def rasterize_worker(pdfpath, pagelimit, size_queue, callback):
    info = pdf2image.pdfinfo_from_path(pdfpath)
    aspect = parse_aspect_from_pdfinfo(info)
    page = 1
    images = [None] * pagelimit
    window_size = size_queue.get()
    image_size = winsize2rasterargs(window_size, aspect)
    while True:
        while not size_queue.empty():
            # Start over!
            window_size = size_queue.get()
            image_size = winsize2rasterargs(window_size, aspect)
            page = 1
        if page == pagelimit + 1:
            # Got through them all without changing size.
            if images[-1] is not None:
                images2 = images
                images = [None] * pagelimit
                callback(images2, window_size)
            else:
                # Already callbacked and no new resize events since.
                pass
        else:
            with tempfile.TemporaryDirectory() as tempdir:
                paths = pdf2image.convert_from_path(
                    pdfpath,
                    size=image_size,
                    output_folder=tempdir,
                    # Do not bother loading as PIL images. Let Pyglet handle loading.
                    # TODO: Try to keep everything in memory.
                    first_page=page,
                    last_page=page,
                    paths_only=True,
                )
                assert len(paths) == 1
                # TODO: Why is pyglet's image loading so slow?
                images[page - 1] = [pyglet.image.load(p) for p in paths][0]
                page += 1


class BlockingRasterizer:
    def __init__(self, path, pagelimit=None):
        self.path = path
        self.pagelimit = pagelimit
        self.images = None
        self.window_size = None
        self.queue = multiprocessing.Queue()
        self.thread = threading.Thread(
            target=rasterize_worker,
            args=(path, pagelimit, self.queue, self.images_done),
        )
        self.lock = threading.Lock()
        self.thread.start()

    def images_done(self, images, window_size):
        with self.lock:
            self.images = images
            self.window_size = window_size

    def push_resize(self, width, height):
        self.queue.put((width, height))

    def draw(self, cursor):
        with self.lock:
            if self.images is None:
                return
            w, h = self.window_size
            image = self.images[cursor]
        dx = (w - image.width) // 2
        dy = (h - image.height) // 2
        # TODO: Get rid of 1-pixel slop.
        assert (dx <= 1) or (dy <= 1)
        image.blit(dx, dy)


def parse_aspect_from_pdfinfo(info):
    size_str = info["Page size"]
    width, _, height, _ = size_str.split(" ")
    return float(width) / float(height)


def main():

    display = pyglet.canvas.get_display()
    screens = display.get_screens()
    modes = [bestmode(s) for s in screens]

    win_audience = pyglet.window.Window(
        caption="audience",
        resizable=True,
    )
    win_presenter = pyglet.window.Window(
        caption="presenter",
        resizable=True,
    )

    path = sys.argv[1]
    info = pdf2image.pdfinfo_from_path(path)
    npages = info["Pages"]
    npages = min(npages, 5)
    rasterizer_audience = BlockingRasterizer(path, pagelimit=npages)
    rasterizer_presenter = BlockingRasterizer(path, pagelimit=npages)

    cursor = Cursor(npages)

    # TODO: Figure out the fine points of pyglet event so we don't need all
    # this copy-paste code.

    @win_audience.event
    def on_resize(width, height):
        nonlocal rasterizer_audience
        print(f"audience resize to {width}, {height}")
        rasterizer_audience.push_resize(width, height)

    @win_presenter.event
    def on_resize(width, height):
        nonlocal rasterizer_presenter
        print(f"presenter resize to {width}, {height}")
        rasterizer_presenter.push_resize(width, height)

    @win_audience.event
    def on_draw():
        win_audience.clear()
        rasterizer_audience.draw(cursor.cursor)
        return pyglet.event.EVENT_HANDLED

    @win_presenter.event
    def on_draw():
        win_presenter.clear()
        if cursor.cursor + 1 < cursor.nslides:
            rasterizer_presenter.draw(cursor.cursor + 1)
        return pyglet.event.EVENT_HANDLED

    def on_tick(dt, keyboard):
        nonlocal cursor
        forward = any(keyboard[k] for k in KEYS_FWD)
        reverse = any(keyboard[k] for k in KEYS_REV)
        if cursor.tick(dt, reverse, forward):
            win_audience.dispatch_event("on_draw")
            win_presenter.dispatch_event("on_draw")

    keyboard = pyglet.window.key.KeyStateHandler()
    win_presenter.push_handlers(keyboard)
    pyglet.clock.schedule_interval(on_tick, 0.05, keyboard=keyboard)

    # Main loop.
    win_presenter.activate()
    pyglet.app.run()



if __name__ == "__main__":
    main()
