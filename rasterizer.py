"""Threaded interruptible PDF rasterizer."""

import multiprocessing
import threading
import queue

import pdf2image


def _winsize2rasterargs(window_size, aspect):
    width, height = window_size
    window_aspect = float(width) / height
    if window_aspect >= aspect:
        width = None
    else:
        height = None
    return (width, height)


def _rasterize_worker(pdfpath, pagelimit, size_queue, callback):
    """Threaded interruptible PDF rasterizer.

    Listens on size_queue for (width, height) tuples representing window resize
    events. When an event arrives, discards any in-progress rasterization and
    starts over. Calls the callback on its own thread when the images for the
    entire PDF are complete and the size has not changed during rasterization.

    Args:
        pdfpath (str): Path of PDF file.
        pagelimit (int): Read this many pages from the file. (Mostly for
            development purposes to keep load time down.)
        size_queue (queue-like): Queue to monitor for size changes.
        callback (fn void(list of PIL images)): Function to call when the full
            PDF is ready.
    """
    info = pdf2image.pdfinfo_from_path(pdfpath)
    aspect = _parse_aspect_from_pdfinfo(info)
    # The (one-based) index of the page we are to rasterize next. If it exceeds
    # page_limit, we have no work to do.
    page = 1
    images = [None] * pagelimit
    # Block indefinitely for first size.
    window_size = size_queue.get()
    image_size = _winsize2rasterargs(window_size, aspect)
    CHUNK = 32
    while True:
        # Get freshest item in queue. This loop would not be necessary if it
        # was possible for a Queue with a maxsize to discard old items instead
        # of blocking when it's full and put() is called.
        try:
            while True:
                window_size = size_queue.get(timeout=0.1)
                if window_size is None:
                    return
                image_size = _winsize2rasterargs(window_size, aspect)
                page = 1
        except queue.Empty:
            pass
        if page == pagelimit + 1:
            # Got through them all without changing size.
            if images[-1] is not None:
                images2 = images
                images = [None] * pagelimit
                callback(images2)
            else:
                # Already callbacked and no new resize events since.
                pass
        else:
            # pdf2image convert_from_bytes just writes to a file, so it's
            # useless for performance.
            chunk = pdf2image.convert_from_path(
                pdfpath,
                thread_count=min(CHUNK, 8),
                size=image_size,
                first_page=page,
                last_page=page+CHUNK-1,
            )
            for i, img in enumerate(chunk):
                images[page - 1] = img
                page += 1


class ThreadedRasterizer:
    """Shared state for communicating with _rasterize_worker thread.

    Also implements the behavior of showing a black slide for an out-of-bounds
    index instead of crashing. In a larger program this should probably be a
    separate layer between the rasterizer and the platform-specific GUI. For
    now it goes here to keep the Pyglet-specific layer as thin as possible.
    """
    def __init__(self, path, pagelimit=None):
        self.images = None
        self.black = None

        self.queue = multiprocessing.Queue()
        self.thread = threading.Thread(
            target=_rasterize_worker,
            args=(path, pagelimit, self.queue, self.images_done),
        )
        self.lock = threading.Lock()

        self.thread.start()

    def images_done(self, images):
        # Defer converting PIL to Pyglet to the GUI thread, otherwise weird
        # things happen with Pyglet deleting textures that are still in use.
        with self.lock:
            self.images = images
            lut = [0] * (256 * 3)
            self.black = images[0].point(lut)

    def push_resize(self, width, height):
        self.queue.put((width, height))

    def get(self, index):
        with self.lock:
            if self.images is None:
                return None
            if index >= 0 and index < len(self.images):
                return self.images[index]
            return self.black

    def shutdown(self):
        self.queue.put(None)
        self.thread.join()


def _parse_aspect_from_pdfinfo(info):
    size_str = info["Page size"]
    width, _, height, _ = size_str.split(" ")
    return float(width) / float(height)
