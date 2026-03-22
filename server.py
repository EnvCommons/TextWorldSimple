from openreward.environments import Server

from textworld_simple import TextWorldSimple

if __name__ == "__main__":
    server = Server([TextWorldSimple])
    server.run()
