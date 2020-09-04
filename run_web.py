
from app_face_recognition import app

if __name__ == "__main__":
    app.run(debug=False, host=app.config['ip_server'], port=app.config['port'], ssl_context='adhoc')