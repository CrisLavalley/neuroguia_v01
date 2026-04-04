from __future__ import annotations
from typing import Any

def require_login() -> dict[str, Any]:
    return {"is_logged_in": False, "email": "", "name": "Usuario"}
