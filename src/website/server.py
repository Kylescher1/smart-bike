from flask import Flask, render_template_string, request
import socket

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string("""
        <html>
            <head><title>Local Web Server</title></head>
            <body>
                <h1>Hello from {{ host }}!</h1>
                <p>Your IP: {{ client_ip }}</p>
                <form action="/echo" method="post">
                    <input name="msg" placeholder="Type a message"/>
                    <button type="submit">Send</button>
                </form>
            </body>
        </html>
    """, host=socket.gethostname(), client_ip=request.remote_addr)

@app.route("/echo", methods=["POST"])
def echo():
    msg = request.form.get("msg", "")
    return f"<p>You said: {msg}</p><a href='/'>Go back</a>"

if __name__ == "__main__":
    # Get your local IP (e.g. 192.168.x.x)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"Server running on http://{local_ip}:5000")
    # host='0.0.0.0' makes it accessible to others on your network
    app.run(host="0.0.0.0", port=5000)
