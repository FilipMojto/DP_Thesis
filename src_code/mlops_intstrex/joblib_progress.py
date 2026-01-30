import joblib
from contextlib import contextmanager

from src_code.mlops_intstrex.reporters.progress_reporter import ProgressReporter



@contextmanager
def joblib_progress(reporter: ProgressReporter, total_tasks):
    """
    A context manager that intercepts joblib's parallel execution to track
    and report progress of background tasks.

    It temporarily modifies the 'BatchCompletionCallBack' and 'Parallel' classes
    within the joblib library to trigger updates to a custom ProgressReporter
    whenever a processing batch completes.

    Args:
        reporter (ProgressReporter): _description_
        total_tasks (_type_): _description_

    Returns:
        _type_: _description_
    """
    # This is the secret sauce: joblib uses this class to handle task completion
    from joblib.parallel import BatchCompletionCallBack

    old_callback = BatchCompletionCallBack.__call__

    def custom_callback(self, *args, **kwargs):
        """
        A replacement for joblib's internal callback function.

        This function is executed in the main process every time a worker
        finishes a 'batch' of work. It calculates how many tasks were in
        that batch and notifies the reporter to advance the progress bar.
        """

        # self.n_completed is the attribute joblib updates
        # We trigger the reporter based on the number of samples/tasks in the batch
        if reporter:
            reporter.advance(self.batch_size)
        return old_callback(self, *args, **kwargs)

    # Monkeypatch the callback
    BatchCompletionCallBack.__call__ = custom_callback

    # We also still patch Parallel just in case,
    # but ensure we force verbose > 0 so joblib actually triggers progress logic
    old_parallel = joblib.parallel.Parallel

    class ProgressParallel(joblib.parallel.Parallel):
        """
        A wrapper around joblib.Parallel that ensures internal progress
        tracking logic is always activated.
        """

        def __init__(self, *args, **kwargs):
            # joblib only calls print_progress if verbose > 0
            # joblib's internal reporting logic is 'lazy'—if verbose is 0,
            # it often skips the code paths we are trying to hook into. 
            # So we force it to be at least 1.
            if kwargs.get("verbose") == 0:
                kwargs["verbose"] = 1
            super().__init__(*args, **kwargs)

    joblib.parallel.Parallel = ProgressParallel

    try:
        yield
    finally:
        BatchCompletionCallBack.__call__ = old_callback
        joblib.parallel.Parallel = old_parallel
