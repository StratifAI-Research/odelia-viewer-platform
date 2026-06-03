"""Tests for shared/timing_utils.py — time_operation context manager."""
import logging
import time

import pytest


def test_time_operation_yields_control():
    """Code inside the with-block executes normally."""
    from shared.timing_utils import time_operation

    logger = logging.getLogger('test')
    executed = []

    with time_operation('test_op', logger):
        executed.append(True)

    assert executed == [True]


def test_time_operation_logs_timing_message(caplog):
    """Logger receives a TIMING message after the block completes."""
    from shared.timing_utils import time_operation

    logger = logging.getLogger('timing_test')

    with caplog.at_level(logging.INFO, logger='timing_test'):
        with time_operation('my_operation', logger):
            pass

    timing_records = [r for r in caplog.records if 'TIMING' in r.message]
    assert len(timing_records) == 1
    assert 'my_operation' in timing_records[0].message


def test_time_operation_log_contains_ms(caplog):
    """The timing log message mentions milliseconds (ms)."""
    from shared.timing_utils import time_operation

    logger = logging.getLogger('ms_test')

    with caplog.at_level(logging.INFO, logger='ms_test'):
        with time_operation('op_with_ms', logger):
            pass

    timing_records = [r for r in caplog.records if 'TIMING' in r.message]
    assert 'ms' in timing_records[0].message


def test_time_operation_elapsed_is_non_negative(caplog):
    """The elapsed value reported must be >= 0."""
    from shared.timing_utils import time_operation
    import re

    logger = logging.getLogger('elapsed_test')

    with caplog.at_level(logging.INFO, logger='elapsed_test'):
        with time_operation('elapsed_op', logger):
            pass

    msg = caplog.records[-1].message
    match = re.search(r'([\d.]+)ms', msg)
    assert match is not None
    elapsed = float(match.group(1))
    assert elapsed >= 0.0


def test_time_operation_logs_even_on_exception(caplog):
    """TIMING is logged even when the body raises an exception (finally block)."""
    from shared.timing_utils import time_operation

    logger = logging.getLogger('exc_test')

    with caplog.at_level(logging.INFO, logger='exc_test'):
        with pytest.raises(RuntimeError):
            with time_operation('failing_op', logger):
                raise RuntimeError('boom')

    timing_records = [r for r in caplog.records if 'TIMING' in r.message]
    assert len(timing_records) == 1
    assert 'failing_op' in timing_records[0].message


def test_time_operation_measures_realistic_duration(caplog):
    """A small sleep is reflected in the timing value > 0."""
    from shared.timing_utils import time_operation

    logger = logging.getLogger('duration_test')

    with caplog.at_level(logging.INFO, logger='duration_test'):
        with time_operation('sleep_op', logger):
            time.sleep(0.01)

    import re
    msg = caplog.records[-1].message
    match = re.search(r'([\d.]+)ms', msg)
    assert match is not None
    assert float(match.group(1)) > 5.0  # at least 5 ms
