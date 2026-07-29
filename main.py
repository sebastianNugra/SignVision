"""
SignVision entry point.

Responsible for bootstrapping and starting the application.
"""

from signvision.app import Application


def main() -> None:
    app = Application()

    try:
        app.initialize()
        app.run()

    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
