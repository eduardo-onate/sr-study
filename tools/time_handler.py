from datetime import timedelta

class TimeHandler:
    """
    A utility class for handling and converting time formats.
    """

    @staticmethod
    def time_to_seconds(time_str: str) -> float:
        """
        Converts a time string to seconds.
        
        Parameters:
        - time_str: Time string in the format "HH:MM:SS" or "HH:MM:SS,mmm".
        
        Returns:
        - Total seconds as a float.
        """
        if ',' in time_str:  # Format with milliseconds
            hours, minutes, seconds, milliseconds = map(int, time_str.split(':'))
            td = timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
        else:  # Format without milliseconds
            hours, minutes, seconds = map(int, time_str.split(':'))
            td = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return td.total_seconds()

    @staticmethod
    def seconds_to_time(seconds: float, include_milliseconds: bool = False) -> str:
        """
        Converts seconds to a time string.
        
        Parameters:
        - seconds: Time in seconds.
        - include_milliseconds: Whether to include milliseconds in the output.
        
        Returns:
        - Time string in the format "HH:MM:SS" or "HH:MM:SS,mmm" if include_milliseconds is True.
        """
        hours, rem = divmod(int(seconds), 3600)
        minutes, seconds = divmod(rem, 60)

        if include_milliseconds:
            milliseconds = int((seconds % 1) * 1000)
            return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"
        return f"{hours:02}:{minutes:02}:{int(seconds):02}"

    @staticmethod
    def duration_between_times(start_time: float, end_time: float, exact_duration: bool = False) -> int:
        """
        Calculates the duration between two times.
        
        Parameters:
        - start_time: Start time in seconds.
        - end_time: End time in seconds.
        - exact_duration: Whether to return the exact duration or the rounded duration.
        
        Returns:
        - Duration in seconds.
        """
        if exact_duration:
            return end_time - start_time
        return int(end_time) - int(start_time)