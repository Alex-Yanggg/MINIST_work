"""
入口：保留历史兼容，通过 mnist_service 暴露的 create_app 构建 FastAPI。
运行：
    uvicorn web_service:app --host 0.0.0.0 --port 8000
"""
import sys 
sys.path.append("../..") 
sys.path.append("../training")


from service import create_app  # noqa: E402
import uvicorn

app = create_app()

if __name__ == "__main__":
    uvicorn.run("scripts.deployment.app:app", host="0.0.0.0", port=8000, reload=False)

