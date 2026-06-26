# Adding New Users to the System

This guide explains how to add new users to the ODELIA deployment.  
Authentication is handled via **Keycloak** using:

- **Realm:** `ohif`
- **Client:** `ohif_viewer`
- **Role assignment:** handled automatically by the realm configuration

These instructions work regardless of whether your deployment uses a domain name, an IP address, HTTP, HTTPS, or a custom port.

---

## 1. Log in to Keycloak

Open your Keycloak admin interface, for example:

- `http://<host>/keycloak/`
- `http://<ip>:<port>/keycloak/`
- `https://<host>/keycloak/`

Log in using your admin credentials (default are admin/admin).

---

## 2. Select the Realm

Top-left dropdown → **`ohif`**

---

## 3. Add a New User

1. Go to **Users** → **Add user**
2. Fill in:
   - **Username** (required)
   - Email (optional)
   - First name / Last name (optional)
   - Ensure **Enabled = ON**
3. Click **Create**

---

## 4. Set Password

1. Open the **Credentials** tab
2. Enter a password
3. Set **Temporary = OFF** unless you want the user to reset it on first login
4. Click **Set Password**

---

## 5. Verify User Access

Open the OHIF viewer using your deployment address, for example:

- `http://<host>/`
- `http://<ip>:<port>/`
- `https://<host>/`

Log in using the newly created username and password.

---

## Troubleshooting

### User cannot log in
- User is not **Enabled**
- Password is not set correctly
- Wrong realm selected during login (`ohif`)
- Typo in username
