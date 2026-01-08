"""
Tests for opticalib.ground.logger module.
"""

import pytest
import os
import logging
import tempfile
import shutil
from opticalib.ground import logger


class TestSetUpLogger:
    """Test set_up_logger function."""

    def test_set_up_logger_creation(self, temp_dir, monkeypatch):
        """Test that logger is created correctly."""
        # Mock the LOGGING_ROOT_FOLDER in the root module
        monkeypatch.setattr("opticalib.core.root.LOGGING_ROOT_FOLDER", temp_dir)
        # Reload the logger module to pick up the new path
        import importlib

        importlib.reload(logger)

        log_file = "test.log"
        test_logger = logger.set_up_logger(log_file, logging.INFO)

        assert isinstance(test_logger, logging.Logger)
        assert test_logger.level == logging.INFO

        # Check that log file was created
        log_path = os.path.join(temp_dir, log_file)
        assert os.path.exists(log_path)

    def test_set_up_logger_default_level(self, temp_dir, monkeypatch):
        """Test logger with default logging level."""
        monkeypatch.setattr("opticalib.core.root.LOGGING_ROOT_FOLDER", temp_dir)
        import importlib

        importlib.reload(logger)

        log_file = "test_default.log"
        test_logger = logger.set_up_logger(log_file)

        assert isinstance(test_logger, logging.Logger)
        assert test_logger.level == logging.DEBUG

    def test_set_up_logger_rotating(self, temp_dir, monkeypatch):
        """Test that logger uses rotating file handler."""
        monkeypatch.setattr("opticalib.core.root.LOGGING_ROOT_FOLDER", temp_dir)
        import importlib

        importlib.reload(logger)

        log_file = "test_rotating.log"
        test_logger = logger.set_up_logger(log_file, logging.INFO)

        # Check that handler is a RotatingFileHandler
        handlers = test_logger.handlers
        assert len(handlers) > 0
        # At least one handler should be a RotatingFileHandler
        rotating_handlers = [
            h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(rotating_handlers) > 0


class TestSystemLogger:
    """Test SystemLogger class."""

    def test_system_logger_init(self):
        """Test SystemLogger initialization."""
        sys_logger = logger.SystemLogger()
        assert sys_logger.logger is not None
        assert isinstance(sys_logger.logger, logging.Logger)

    def test_system_logger_with_class(self):
        """Test SystemLogger initialization with a class."""
        class TestClass:
            pass

        sys_logger = logger.SystemLogger(TestClass)
        assert sys_logger.the_class == TestClass

    def test_system_logger_info(self, caplog):
        """Test SystemLogger info method."""
        sys_logger = logger.SystemLogger()
        with caplog.at_level(logging.INFO):
            sys_logger.info("Info message")
            assert "Info message" in caplog.text

    def test_system_logger_debug(self, caplog):
        """Test SystemLogger debug method."""
        sys_logger = logger.SystemLogger()
        with caplog.at_level(logging.DEBUG):
            sys_logger.debug("Debug message")
            assert "Debug message" in caplog.text

    def test_system_logger_warning(self, caplog):
        """Test SystemLogger warning method."""
        sys_logger = logger.SystemLogger()
        with caplog.at_level(logging.WARNING):
            sys_logger.warning("Warning message")
            assert "Warning message" in caplog.text

    def test_system_logger_error(self, caplog):
        """Test SystemLogger error method."""
        sys_logger = logger.SystemLogger()
        with caplog.at_level(logging.ERROR):
            sys_logger.error("Error message")
            assert "Error message" in caplog.text

    def test_system_logger_critical(self, caplog):
        """Test SystemLogger critical method."""
        sys_logger = logger.SystemLogger()
        with caplog.at_level(logging.CRITICAL):
            sys_logger.critical("Critical message")
            assert "Critical message" in caplog.text

    def test_system_logger_log_with_level(self, caplog):
        """Test SystemLogger log method with explicit level."""
        sys_logger = logger.SystemLogger()
        with caplog.at_level(logging.DEBUG):
            sys_logger.log(message="Debug message", level="DEBUG")
            assert "Debug message" in caplog.text

    def test_system_logger_log_with_class_name(self, caplog):
        """Test SystemLogger log method includes class name."""
        class TestClass:
            pass

        sys_logger = logger.SystemLogger(TestClass)
        with caplog.at_level(logging.INFO):
            sys_logger.info("Test message")
            assert "TestClass" in caplog.text
            assert "Test message" in caplog.text

    def test_system_logger_log_without_class_name(self, caplog):
        """Test SystemLogger log method without class name."""
        class TestClass:
            pass

        sys_logger = logger.SystemLogger(TestClass)
        with caplog.at_level(logging.INFO):
            sys_logger.log(message="Test message", level="INFO", no_class=True)
            assert "TestClass" not in caplog.text
            assert "Test message" in caplog.text


class TestLogFunction:
    """Test log function."""

    def test_log_function_debug(self, caplog):
        """Test logging at DEBUG level."""
        test_logger = logger.SystemLogger.getSystemLogger()
        with caplog.at_level(logging.DEBUG):
            logger.log(test_logger, message="Debug message", level="DEBUG")
            assert "Debug message" in caplog.text

    def test_log_function_info(self, caplog):
        """Test logging at INFO level."""
        test_logger = logger.SystemLogger.getSystemLogger()
        with caplog.at_level(logging.INFO):
            logger.log(test_logger, message="Info message", level="INFO")
            assert "Info message" in caplog.text

    def test_log_function_warning(self, caplog):
        """Test logging at WARNING level."""
        test_logger = logger.SystemLogger.getSystemLogger()
        with caplog.at_level(logging.WARNING):
            logger.log(test_logger, message="Warning message", level="WARNING")
            assert "Warning message" in caplog.text

    def test_log_function_error(self, caplog):
        """Test logging at ERROR level."""
        test_logger = logger.SystemLogger.getSystemLogger()
        with caplog.at_level(logging.ERROR):
            logger.log(test_logger, message="Error message", level="ERROR")
            assert "Error message" in caplog.text

    def test_log_function_critical(self, caplog):
        """Test logging at CRITICAL level."""
        test_logger = logger.SystemLogger.getSystemLogger()
        with caplog.at_level(logging.CRITICAL):
            logger.log(test_logger, message="Critical message", level="CRITICAL")
            assert "Critical message" in caplog.text

    def test_log_function_lowercase(self, caplog):
        """Test logging with lowercase level."""
        test_logger = logger.SystemLogger.getSystemLogger()
        with caplog.at_level(logging.INFO):
            logger.log(test_logger, message="Info message", level="info")
            assert "Info message" in caplog.text

    def test_log_function_invalid_level(self, caplog):
        """Test logging with invalid level defaults to DEBUG."""
        test_logger = logger.SystemLogger.getSystemLogger()
        with caplog.at_level(logging.DEBUG):
            logger.log(test_logger, message="Invalid level message", level="INVALID")
            assert "Invalid level message" in caplog.text
            assert "Invalid log level" in caplog.text

    def test_log_function_default_level(self, caplog):
        """Test logging with default level (INFO)."""
        test_logger = logger.SystemLogger.getSystemLogger()
        with caplog.at_level(logging.INFO):
            logger.log(test_logger, message="Default level message")
            assert "Default level message" in caplog.text

    def test_log_function_with_class(self, caplog):
        """Test log function with class name."""
        test_logger = logger.SystemLogger.getSystemLogger()

        class TestClass:
            pass

        with caplog.at_level(logging.INFO):
            logger.log(test_logger, message="Test message", the_class=TestClass)
            assert "TestClass" in caplog.text
            assert "Test message" in caplog.text


class TestTxtLogger:
    """Test txtLogger class."""

    def test_txt_logger_init(self, temp_dir):
        """Test txtLogger initialization."""
        log_file = os.path.join(temp_dir, "test.txt")
        txt_log = logger.txtLogger(log_file)

        assert txt_log.file_path == log_file

    def test_txt_logger_log(self, temp_dir):
        """Test txtLogger log method."""
        log_file = os.path.join(temp_dir, "test.txt")
        txt_log = logger.txtLogger(log_file)

        message = "Test log message"
        txt_log.log(message)

        # Check that file was created and contains the message
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            assert message in content

    def test_txt_logger_multiple_logs(self, temp_dir):
        """Test txtLogger with multiple log entries."""
        log_file = os.path.join(temp_dir, "test.txt")
        txt_log = logger.txtLogger(log_file)

        messages = ["Message 1", "Message 2", "Message 3"]
        for msg in messages:
            txt_log.log(msg)

        # Check that all messages are in the file
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            for msg in messages:
                assert msg in content

    def test_txt_logger_append(self, temp_dir):
        """Test that txtLogger appends to existing file."""
        log_file = os.path.join(temp_dir, "test.txt")
        txt_log = logger.txtLogger(log_file)

        txt_log.log("First message")
        txt_log.log("Second message")

        # Check that both messages are in the file
        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert "First message" in lines[0]
            assert "Second message" in lines[1]
