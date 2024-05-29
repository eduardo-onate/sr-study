import time
from tqdm import tqdm

class ProgressBarHandler:
    """
    A utility to visualize task progress using a progress bar.
    """
    
    _active_instance = None  # To ensure only one active progress bar at a time

    def __init__(self, total: int, description: str = ''):
        """
        Initializes the ProgressBarHandler object with the total progress and a description.
        
        Parameters:
        - total: Total progress to be achieved.
        - description: Description of the progress bar.
        """
        self.total = total
        self.description = description
        self.tqdm_instance = None
        self._progress_so_far = 0  
        self.start_time = None

    def open(self):
        """
        Opens the progress bar. If another progress bar is active, it stops it.
        """
        if ProgressBarHandler._active_instance is self:
            return
        if ProgressBarHandler._active_instance:
            ProgressBarHandler._active_instance.stop()
        ProgressBarHandler._active_instance = self

        if not self.start_time:
            self.start_time = time.time()
        
        elapsed_time = time.time() - self.start_time
        self.tqdm_instance = tqdm(total=100.0, desc=self.description, unit='%', 
                                  initial=round((self._progress_so_far * 100 / self.total), 1), 
                                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
        self.tqdm_instance.start_t = self.start_time - elapsed_time

    def update(self, absolute_increment: int):
        """
        Updates the progress bar with the given increment.
        
        Parameters:
        - absolute_increment: The amount by which the progress should be incremented.
        """
        self._progress_so_far += absolute_increment
        percentage_progress = round((self._progress_so_far / self.total) * 100, 1)
        if self.tqdm_instance:
            self.tqdm_instance.n = min(percentage_progress, 100)
            self.tqdm_instance.refresh()

    def stop(self):
        """
        Stops the progress bar without closing it.
        """
        if self.tqdm_instance:
            self._progress_so_far = (self.tqdm_instance.n / 100) * self.total
            self.tqdm_instance.close()
            self.tqdm_instance = None
        if ProgressBarHandler._active_instance is self:
            ProgressBarHandler._active_instance = None

    def close(self):
        """
        Closes the progress bar.
        """
        if self.tqdm_instance:
            self.tqdm_instance.n = 100
            self.tqdm_instance.refresh()
            self.tqdm_instance.close()
            self.tqdm_instance = None
        if ProgressBarHandler._active_instance is self:
            ProgressBarHandler._active_instance = None