from app import app, db
from app import User, Prediction

with app.app_context():
    print("Creating database tables...")
    db.create_all()
    print("Tables created successfully.")
