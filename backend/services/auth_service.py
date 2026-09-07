"""
BuyWise — User Account, Authentication & Wishlist Service
==========================================================
Simple, secure authentication using SHA-256 salted hashes and bearer tokens.
Manages user profiles, saved wishlists, and recently viewed products.
"""

import sqlite3
import os
import hashlib
import secrets
import json
from typing import Dict, Any, Optional, List, Union

DB_PATH = os.path.join(os.path.dirname(__file__), '../buywise.db')
SALT = "buywise_2026_salt_secret_"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    return hashlib.sha256((SALT + password).encode()).hexdigest()

def generate_token(user_id: int, email: str) -> str:
    payload = f"{user_id}:{email}:{secrets.token_hex(8)}"
    return hashlib.sha256(payload.encode()).hexdigest() + f".{user_id}"

def get_user_id_from_token(token: str) -> Optional[int]:
    if not token or "." not in token:
        return None
    try:
        user_id = int(token.split(".")[-1])
        return user_id
    except ValueError:
        return None

class AuthService:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def signup(self, email: str, username: str, password: str) -> Dict[str, Any]:
        email = email.strip().lower()
        username = username.strip()
        if not email or not username or not password:
            return {"success": False, "error": "Email, username, and password are required."}
        if len(password) < 6:
            return {"success": False, "error": "Password must be at least 6 characters long."}

        pwd_hash = hash_password(password)
        conn = get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (email, username, password_hash) VALUES (?, ?, ?)",
                (email, username, pwd_hash)
            )
            conn.commit()
            user_id = cursor.lastrowid
            token = generate_token(user_id, email)
            return {
                "success": True,
                "user": {"id": user_id, "email": email, "username": username},
                "token": token
            }
        except sqlite3.IntegrityError:
            return {"success": False, "error": "An account with this email already exists."}
        finally:
            conn.close()

    def login(self, email: str, password: str) -> Dict[str, Any]:
        email = email.strip().lower()
        pwd_hash = hash_password(password)
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, username, password_hash FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if not user or user["password_hash"] != pwd_hash:
            return {"success": False, "error": "Invalid email or password."}

        token = generate_token(user["id"], email)
        return {
            "success": True,
            "user": {"id": user["id"], "email": user["email"], "username": user["username"]},
            "token": token
        }

    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, username, created_at FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return None

        # Fetch count of saved items
        cursor.execute("SELECT COUNT(*) FROM user_saved_products WHERE user_id = ?", (user_id,))
        saved_count = cursor.fetchone()[0]

        # Fetch count of viewed items
        cursor.execute("SELECT COUNT(*) FROM user_history WHERE user_id = ?", (user_id,))
        history_count = cursor.fetchone()[0]

        conn.close()
        return {
            "id": user["id"],
            "email": user["email"],
            "username": user["username"],
            "created_at": user["created_at"],
            "saved_count": saved_count,
            "history_count": history_count
        }

    def toggle_saved_product(self, user_id: int, product_id: Union[str, int]) -> Dict[str, Any]:
        conn = get_conn()
        cursor = conn.cursor()
        pid_str = str(product_id).strip()
        cursor.execute("SELECT id FROM user_saved_products WHERE user_id = ? AND product_id = ?", (user_id, pid_str))
        exists = cursor.fetchone()
        if exists:
            cursor.execute("DELETE FROM user_saved_products WHERE user_id = ? AND product_id = ?", (user_id, pid_str))
            is_saved = False
        else:
            cursor.execute("INSERT OR IGNORE INTO user_saved_products (user_id, product_id) VALUES (?, ?)", (user_id, pid_str))
            is_saved = True
        conn.commit()
        conn.close()
        return {"success": True, "product_id": pid_str, "is_saved": is_saved}

    def get_saved_products(self, user_id: int) -> List[Dict[str, Any]]:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pc.canonical_id, pc.product_json, usp.created_at
            FROM user_saved_products usp
            LEFT JOIN product_cache pc ON usp.product_id = pc.canonical_id OR usp.product_id = pc.id
            WHERE usp.user_id = ?
            ORDER BY usp.created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        products = []
        for r in rows:
            if r["product_json"]:
                try:
                    p = json.loads(r["product_json"])
                    p["id"] = r["canonical_id"] or p.get("canonical_id")
                    products.append(p)
                except Exception:
                    pass
        return products

    def record_history(self, user_id: int, product_id: Union[str, int]) -> None:
        conn = get_conn()
        cursor = conn.cursor()
        pid_str = str(product_id).strip()
        cursor.execute("""
            INSERT INTO user_history (user_id, product_id, viewed_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, product_id) DO UPDATE SET viewed_at = CURRENT_TIMESTAMP
        """, (user_id, pid_str))
        conn.commit()
        conn.close()

    def get_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pc.canonical_id, pc.product_json, uh.viewed_at 
            FROM user_history uh
            LEFT JOIN product_cache pc ON uh.product_id = pc.canonical_id OR uh.product_id = pc.id
            WHERE uh.user_id = ?
            ORDER BY uh.viewed_at DESC
            LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        conn.close()
        
        products = []
        for r in rows:
            if r["product_json"]:
                try:
                    p = json.loads(r["product_json"])
                    p["id"] = r["canonical_id"] or p.get("canonical_id")
                    p["viewed_at"] = r["viewed_at"]
                    products.append(p)
                except Exception:
                    pass
        return products
