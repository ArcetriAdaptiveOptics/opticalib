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
        rotating_handlers = [h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(rotating_handlers) > 0


class TestLog:
    """Test log function."""

    def test_log_debug(self, caplog):
        """Test logging at DEBUG level."""
        with caplog.at_level(logging.DEBUG):
            logger.log("Debug message", "DEBUG")
            assert "Debug message" in caplog.text

    def test_log_info(self, caplog):
        """Test logging at INFO level."""
        with caplog.at_level(logging.INFO):
            logger.log("Info message", "INFO")
            assert "Info message" in caplog.text

    def test_log_warning(self, caplog):
        """Test logging at WARNING level."""
        with caplog.at_level(logging.WARNING):
            logger.log("Warning message", "WARNING")
            assert "Warning message" in caplog.text

    def test_log_error(self, caplog):
        """Test logging at ERROR level."""
        with caplog.at_level(logging.ERROR):
            logger.log("Error message", "ERROR")
            assert "Error message" in caplog.text

    def test_log_critical(self, caplog):
        """Test logging at CRITICAL level."""
        with caplog.at_level(logging.CRITICAL):
            logger.log("Critical message", "CRITICAL")
            assert "Critical message" in caplog.text

    def test_log_lowercase(self, caplog):
        """Test logging with lowercase level."""
        with caplog.at_level(logging.INFO):
            logger.log("Info message", "info")
            assert "Info message" in caplog.text

    def test_log_invalid_level(self, caplog):
        """Test logging with invalid level defaults to DEBUG."""
        with caplog.at_level(logging.DEBUG):
            logger.log("Invalid level message", "INVALID")
            assert "Invalid level message" in caplog.text
            assert "Invalid log level" in caplog.text

    def test_log_default_level(self, caplog):
        """Test logging with default level (INFO)."""
        with caplog.at_level(logging.INFO):
            logger.log("Default level message")
            assert "Default level message" in caplog.text


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

