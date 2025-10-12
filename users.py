from app import app, db, User
from werkzeug.security import generate_password_hash

# This script is for creating pre-defined users.
# Run it from your terminal: python create_users.py

def create_user(username, password):
    with app.app_context():
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            print(f"User '{username}' already exists.")
            return

        # Hash the password for secure storage
        hashed_password = generate_password_hash(password)

        # Create the new user
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        print(f"✅ User '{username}' created successfully.")

if __name__ == '__main__':
    # --- ADD YOUR USERS HERE ---
    # Format: create_user('username', 'password')
    create_user('admin', 'securepassword123')
    create_user('farmer_joe', 'plantsrule')
    create_user('botanist_bob', 'leaflover2023')
    
    print("\nUser creation process finished.")
