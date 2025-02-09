import threading
import time
import logging
from flask import Flask, render_template

from service_bus import *

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/daserver')
def daserver():
    """
    Serves as the main entry point for the server.
    """
    return render_template('index.html')

if __name__ == '__main__':
    # Start the message receiver in a separate thread
    receiver_thread = threading.Thread(target=receive_messages_from_queue, daemon=True)
    receiver_thread.start()

    # Run the Flask app
    app.run(debug=False)
