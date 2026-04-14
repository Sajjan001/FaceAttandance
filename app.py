import streamlit as st
import cv2
import face_recognition
import numpy as np
import faiss
import pickle
import os
import time
import pandas as pd
from datetime import datetime
import json
import hashlib

# ==========================================
# ⚙️ Configuration & Database Setup
# ==========================================
# We use a 128-dimensional vector to represent a face's features mathematically.
DIM = 128
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "embeddings.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

def load_database():
    """Loads the face database if it exists, otherwise creates empty ones."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if os.path.exists(INDEX_PATH):
        # Load the FAISS vector index (mathematical mappings of faces)
        index = faiss.read_index(INDEX_PATH)
        # Load the names associated with the vectors
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        # Create a blank vector space if no database exists
        index = faiss.IndexFlatL2(DIM)
        metadata = []
    
    return index, metadata

def save_database(index, metadata):
    """Saves the current known faces and names to the hard drive."""
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

# Initialize database on app startup
index, metadata = load_database()

# ==========================================
# 🔐 Authentication Database Setup (JSON)
# ==========================================
USERS_FILE = "users.json"
ATTENDANCE_FILE = "attendance.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_users():
    if not os.path.exists(USERS_FILE):
        # We start with exactly ONE default admin
        users = {
            "admin@admin.com": {
                "password": hash_password("admin123"),
                "role": "Admin"
            }
        }
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

def init_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            json.dump([], f, indent=4)

def signup_user(email, password, role):
    init_users()
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
    users[email] = {
        "password": hash_password(password),
        "role": role
    }
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def check_login(email, password, role):
    init_users()
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
    if email in users and users[email]["password"] == hash_password(password) and users[email]["role"] == role:
        return True
    return False

def check_email_exists(email):
    init_users()
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
    return email in users

# Initialize both files
init_users()
init_attendance()

# ==========================================
# 🎨 User Interface (Streamlit)
# ==========================================
st.title("🎯 Face Recognition Attendance System")

# Initialize session state for auth
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# If not logged in, show purely the Login flow
if not st.session_state.logged_in:
    st.sidebar.title("Login")
    selected_role = st.sidebar.selectbox("Role", ["Student", "Admin"])

    st.header(f"🔐 Login as {selected_role}")
    
    if selected_role == "Admin":
        st.info("Default Admin ID: `admin@admin.com` | Password: `admin123`")
    else:
        st.info("Students must receive their login credentials directly from an Admin.")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_login(email, password, selected_role):
            st.session_state.logged_in = True
            st.session_state.user_role = selected_role
            st.session_state.user_email = email
            st.rerun()
        else:
            st.error("Incorrect Email, Password, or Role.")

# If logged in, show the App
else:
    st.sidebar.markdown(f"**Logged in as:** `{st.session_state.user_email}`")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.user_email = None
        st.rerun()

    st.sidebar.divider()
    
    role = st.session_state.user_role
    if role == "Admin":
        menu = st.sidebar.radio("Admin Menu", ["Create Student Account", "Add Student Face", "Download Attendance"])
    else:
        menu = st.sidebar.radio("Student Menu", ["Mark Attendance"])


    # ==========================================
    # 0️⃣ Mode: Create Student Account (Admin Only)
    # ==========================================
    if menu == "Create Student Account":
        st.header("👤 Create Student Account")
        st.info("You exclusively hold the power to create student credentials. Distribute these securely to students.")
        
        new_email = st.text_input("Student Email")
        new_password = st.text_input("Student Password") # visible so admin can see what they generate
        
        if st.button("Create Account"):
            if not new_email or not new_password:
                st.warning("Please fill in both email and password")
            elif check_email_exists(new_email):
                st.error("Account already exists!")
            else:
                signup_user(new_email, new_password, "Student")
                st.success(f"Successfully created account for {new_email}! Setup complete.")

    # ==========================================
    # 1️⃣ Mode: Add a New Student (Admin Only)
    # ==========================================
    elif menu == "Add Student Face":
        st.header("➕ Add New Student Face")

        name = st.text_input("Enter Student Name")
        
        st.info("Upload an image or take a picture using the camera.")
        input_method = st.radio("Select Input Method", ("Camera", "File Upload"))
        
        image_file = None
        if input_method == "Camera":
            image_file = st.camera_input("Take a clear picture of the face")
        else:
            image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

        if image_file is not None:
            if name == "":
                st.warning("Please enter a name first.")
            else:
                with st.spinner("Processing face..."):
                    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    faces = face_recognition.face_locations(rgb)

                    if faces:
                        # Add encoding for the primary face
                        encoding = face_recognition.face_encodings(rgb, faces)[0]
                        index.add(np.array([encoding]).astype("float32"))
                        metadata.append(name)

                        save_database(index, metadata)
                        st.success(f"✅ {name}'s facial data was successfully added to the system!")
                    else:
                        st.error("No face detected in the image. Please try again with a clearer photo.")


    # ==========================================
    # 2️⃣ Mode: Recognize Faces & Mark Attendance (Student Only)
    # ==========================================
    elif menu == "Mark Attendance":
        st.header("🎥 Mark Your Attendance")

        st.info("Take a distinct photo of your face to mark your attendance.")
        
        image_file = st.camera_input("Take a picture for attendance")
        
        if image_file is not None:
            with st.spinner("Verifying identity..."):
                file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                faces = face_recognition.face_locations(rgb)
                
                if faces:
                    encodings = face_recognition.face_encodings(rgb, faces)
                    
                    enc = encodings[0]
                    query = np.array([enc]).astype("float32")
                    
                    if index.ntotal > 0:
                        D, I = index.search(query, k=1)
                        
                        if D[0][0] < 0.6:
                            matched_name = metadata[I[0][0]]
                            
                            if "marked_names" not in st.session_state:
                                st.session_state.marked_names = set()
                                
                            if matched_name not in st.session_state.marked_names:
                                init_attendance()
                                with open(ATTENDANCE_FILE, "r") as f:
                                    records = json.load(f)
                                    
                                records.append({
                                    "Name": matched_name,
                                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                                
                                with open(ATTENDANCE_FILE, "w") as f:
                                    json.dump(records, f, indent=4)
                                    
                                st.session_state.marked_names.add(matched_name)
                                st.success(f"✅ Attendance successfully recorded for {matched_name}!")
                            else:
                                st.info(f"Attendance already recorded for {matched_name} this session.")
                        else:
                            st.error("Face not recognized in the database. Please contact Admin.")
                    else:
                        st.error("Database is empty. No faces are currently registered.")
                else:
                    st.error("No face detected. Please ensure your face is clearly visible.")

    # ==========================================
    # 3️⃣ Mode: Download Attendance (Admin Only)
    # ==========================================
    elif menu == "Download Attendance":
        st.header("📄 Download Attendance Records")
        init_attendance()
        
        with open(ATTENDANCE_FILE, "r") as f:
            records = json.load(f)
            
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="attendance.csv",
                mime="text/csv"
            )
        else:
            st.info("No attendance records found.")
