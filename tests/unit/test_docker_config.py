"""Unit and configuration validation tests for Phase 11 Docker containerization."""

from pathlib import Path
import pytest
import yaml


def test_docker_backend_file_structure():
    """Verifies Dockerfile.backend directives, healthcheck, and entrypoint."""
    dockerfile_path = Path("docker/Dockerfile.backend")
    assert dockerfile_path.exists(), "docker/Dockerfile.backend does not exist"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in content
    assert "requirements.txt" in content
    assert "ENTRYPOINT" in content
    assert "HEALTHCHECK" in content
    assert "EXPOSE 8000" in content


def test_docker_frontend_file_structure():
    """Verifies multi-stage build directives in Dockerfile.frontend."""
    dockerfile_path = Path("docker/Dockerfile.frontend")
    assert dockerfile_path.exists(), "docker/Dockerfile.frontend does not exist"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM node:20-alpine AS builder" in content
    assert "FROM nginx:1.27-alpine AS runner" in content
    assert "COPY --from=builder /app/dist /usr/share/nginx/html" in content
    assert "EXPOSE 80" in content
    assert "HEALTHCHECK" in content


def test_nginx_configuration_integrity():
    """Verifies Nginx reverse proxy and SPA routing directives."""
    nginx_path = Path("docker/nginx.conf")
    assert nginx_path.exists(), "docker/nginx.conf does not exist"

    content = nginx_path.read_text(encoding="utf-8")
    assert "proxy_pass http://backend:8000/api/;" in content
    assert "try_files $uri $uri/ /index.html;" in content
    assert "location /healthz" in content


def test_docker_compose_specification():
    """Validates docker-compose.yml service topology, network, and volume configurations."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml does not exist"

    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "db" in services, "Missing db service"
    assert "backend" in services, "Missing backend service"
    assert "frontend" in services, "Missing frontend service"

    # Verify db configuration
    db_service = services["db"]
    assert "postgres:16-alpine" in db_service["image"]
    assert "healthcheck" in db_service
    assert "postgres_data" in str(db_service["volumes"])

    # Verify backend configuration
    backend_service = services["backend"]
    assert backend_service["depends_on"]["db"]["condition"] == "service_healthy"
    assert "healthcheck" in backend_service

    # Verify frontend configuration
    frontend_service = services["frontend"]
    assert frontend_service["depends_on"]["backend"]["condition"] == "service_healthy"

    # Verify networks
    assert "hitl-network" in compose_data.get("networks", {})


def test_dockerignore_security_exclusions():
    """Verifies that secrets, virtual environments, and caches are ignored."""
    dockerignore_path = Path(".dockerignore")
    assert dockerignore_path.exists(), ".dockerignore does not exist"

    content = dockerignore_path.read_text(encoding="utf-8")
    assert ".venv" in content
    assert "__pycache__" in content
    assert ".git" in content
    assert "node_modules" in content
    assert ".env.local" in content


def test_graph_hop_selector_schema_contract():
    """
    Verifies that the graph route schema contract supports 1, 2, and 3 hops.
    """
    from src.module6_api.routers.graph import router
    # Find the subgraph route parameter schema
    for route in router.routes:
        if getattr(route, "path", "") == "/{tx_id}/subgraph":
            # Find hops parameter
            hops_param = next((param for param in route.dependant.query_params if param.name == "hops"), None)
            assert hops_param is not None, "Missing hops query parameter on /{tx_id}/subgraph"
            assert hops_param.field_info.default == 2
            assert hops_param.field_info.ge == 1
            assert hops_param.field_info.le == 3
