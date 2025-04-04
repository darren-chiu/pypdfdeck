"""Threaded interruptible PDF rasterizer."""

from dataclasses import dataclass
import os
import queue
import threading
import time
from typing import Any, Dict

import pdf2image


CHUNK_PAGES = 32
MAX_THREADS = os.cpu_count() - 1
LRU_SIZE = 4

_EXIT_SENTINEL = -1
_IDLE_SENTINEL = -2


# There are pip packages for this, but we try to minimize dependencies.
#
# Implementation note: Theoretically, enough operations self.counter will
# become bignum. But in practice, nobody will resize their window billions of
# times, so we are OK to use this logic instead of the more complex logic
# needed to ensure that age counters are no larger than the cache size. Also,
# we could use a priority queue for O(log N) evicition, but our cache size is
# not big enough for that to matter.
class LRUDict:
    @dataclass
    class _LRUEntry:
        used: int
        item: Any

    def __init__(self, size):
        self.dict: Dict[Any, LRUDict._LRUEntry] = {}
        self.size: int = size
        self.counter: int = 0

    def __contains__(self, key):
        return key in self.dict

    def __getitem__(self, key):
        entry = self.dict[key]
        entry.used = self.counter
        self.counter += 1
        return entry.item

    def __setitem__(self, key, value):
        if (len(self.dict) == self.size) and (key not in self.dict):
            key_oldest, _ = min(self.dict.items(), key=lambda item: item[1].used)
            del self.dict[key_oldest]
        self.dict[key] = LRUDict._LRUEntry(self.counter, value)
        self.counter += 1


def _winsize2rasterargs(window_size, aspect):
    width, height = window_size
    window_aspect = float(width) / height
    if window_aspect >= aspect:
        width = None
    else:
        height = None
    return (width, height)


def _parse_aspect_from_pdfinfo(info):
    size_str = info["Page size"]
    width, _, height, _, _ = size_str.split(" ")
    return float(width) / float(height)


def _rasterize_worker(pdfpath, aspect, pagelimit, size_queue, image_queue):
    """Threaded interruptible PDF rasterizer.

    Listens on size_queue for (width, height) tuples representing window resize
    events. When an event arrives, discards any in-progress rasterization and
    starts over. Calls the callback on its own thread when the images for the
    entire PDF are complete and the size has not changed during rasterization.

    Args:
        pdfpath (str): Path of PDF file.
        aspect (float): Aspect ratio (width/height) of PDF file.
        pagelimit (int): Read this many pages from the file. (Mostly for
            development purposes to keep load time down.)
        size_queue (queue-like): Queue to monitor for size changes.
        image_queue (queue-like): Queue to return completed renders.
    """
    images = [None] * pagelimit

    # Loop invariant: this is the (one-based) index of the page we should
    # rasterize next. If it exceeds page_limit, we have no work to do.
    page = 1

    # Block indefinitely for first size.
    image_size = size_queue.get()

    while True:
        # Get freshest item in size_queue. This loop would not be necessary if
        # it was possible for a Queue with a maxsize to discard old items
        # instead of blocking when it's full and put() is called.
        try:
            while True:
                image_size = size_queue.get(timeout=0.1)
                if image_size == _EXIT_SENTINEL:
                    # Stop the thread and exit cleanly.
                    return
                elif image_size == _IDLE_SENTINEL:
                    # If working, stop.
                    page = pagelimit + 1
                    images = [None] * pagelimit
                else:
                    page = 1
        except queue.Empty:
            pass

        if page == pagelimit + 1:
            # Got through them all without changing size - push exactly once.
            if images[-1] is not None:
                image_queue.put((image_size, images))
                images = [None] * pagelimit
        else:
            # One might hope that pdf2image.convert_from_bytes is faster by
            # staying in-memory, but it just writes the bytes to a temp file.
            chunk = pdf2image.convert_from_path(
                pdfpath,
                thread_count=MAX_THREADS,
                size=image_size,
                first_page=page,
                last_page=page+CHUNK_PAGES-1,
            )
            for img in chunk:
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
        self.render_start_time = None

        self.cache = LRUDict(4)

        info = pdf2image.pdfinfo_from_path(path)
        self.aspect = _parse_aspect_from_pdfinfo(info)

        self.size_queue = queue.Queue()
        self.image_queue = queue.Queue()
        self.thread = threading.Thread(
            target=_rasterize_worker,
            args=(path, self.aspect, pagelimit, self.size_queue, self.image_queue),
        )
        self.thread.start()

    def push_resize(self, w, h):
        if (w, h) not in self.cache:
            self.size_queue.put((w, h))
            self.render_start_time = time.time()
        else:
            # _IDLE_SENTINEL avoids the following bug:
            # 1) resize to non-cached size 1, start a render
            # 2) resize to cached size 2 before render is done
            # 3) render of now-invalid size 1 finishes, is pushed to image_queue
            self.size_queue.put(_IDLE_SENTINEL)
            self._set_images(self.cache[(w, h)])
            print(f"retrieved ({w:.1f}, {h:.1f}) render from cache.")

    def get(self, index):
        try:
            (w, h), images = self.image_queue.get(block=False)
            self.cache[(w, h)] = images
            self._set_images(images)
            duration = time.time() - self.render_start_time
            print(f"rendered ({w:.1f}, {h:.1f}) in {duration:.2f} sec.")
        except queue.Empty:
            pass
        if self.images is None:
            return None
        if 0 <= index < len(self.images):
            return self.images[index]
        return self.black

    def shutdown(self):
        self.size_queue.put(_EXIT_SENTINEL)
        self.thread.join()

    def _set_images(self, images):
        self.images = images
        lut = [0] * (256 * 3)
        self.black = images[0].point(lut)
