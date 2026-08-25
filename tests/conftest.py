"""Shared fixtures for the test suite.

The whole suite runs offline: any test that would otherwise reach the network
either mocks the transport or asserts on the degraded path instead.
"""

import pytest

LEGITIMATE_HTML = """
<html>
  <head>
    <title>Example Shop</title>
    <meta name="description" content="A shop">
    <link rel="icon" href="/favicon.ico">
    <link rel="stylesheet" href="/main.css">
    <style>.a{color:red}</style>
  </head>
  <body>
    <p>&copy; 2026 Example Inc.</p>
    <a href="https://twitter.com/example">Twitter</a>
    <a href="/about">About</a>
    <img src="/logo.png">
    <img src="/hero.png">
    <script src="/app.js"></script>
  </body>
</html>
"""

PHISHING_HTML = """
<html>
  <body>
    <form>
      <input type="password" name="pw">
      <input type="submit" value="Sign in">
    </form>
    <iframe src="http://elsewhere.example"></iframe>
    <script>window.location = "http://elsewhere.example";</script>
    <script>document.location.href = "http://other.example";</script>
  </body>
</html>
"""


@pytest.fixture
def legitimate_html() -> str:
    """HTML resembling an established, legitimate site."""
    return LEGITIMATE_HTML


@pytest.fixture
def phishing_html() -> str:
    """HTML resembling a minimal credential-harvesting page."""
    return PHISHING_HTML
