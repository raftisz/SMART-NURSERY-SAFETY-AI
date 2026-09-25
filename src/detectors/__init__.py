"""Detection module adapters. Each wraps an existing model behind BaseDetector.

Adapters only report what the model sees. They never import the LINE notifier
or any notification channel (enforced by tests/test_factory.py).
"""
