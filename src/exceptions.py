"""Project exception types. See the Error handling section of AGENTS.md."""


class BaseProjectError(Exception):
    """Base class for every error this project raises on purpose."""


class ConfigurationError(BaseProjectError):
    """A configuration value is invalid."""


class GameEnvironmentError(BaseProjectError):
    """The game environment could not be created or returned something unexpected."""


class TrainingError(BaseProjectError):
    """Training cannot continue, for example after a non-finite loss or gradient."""
