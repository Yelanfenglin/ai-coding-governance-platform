FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

# 依赖层单独缓存：只要 pyproject.toml 不变就不重装。
COPY pyproject.toml ./
COPY app ./app
COPY evaluation ./evaluation
RUN pip install --no-cache-dir ".[web,observability]"

# 会话与工作区都在 /.agent-runtime 下，由 compose 挂载卷持久化。
ENV AGENT_WORKSPACE=/workspace \
    AGENT_ALLOWED_HOSTS=agent-runtime,host.docker.internal

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8081/api/agent/health',timeout=3)"

CMD ["agent-web", "--host", "0.0.0.0", "--port", "8081"]
